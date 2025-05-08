#!/usr/bin/env python3
import os
import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
import optuna

warnings.filterwarnings('ignore')

# === Constants & Parameters ===
LATITUDE          = 32.7
LONGITUDE         = -114.63
PANEL_AREA_TOTAL  = 256_000     # m² for 40 MW solar farm
EFFICIENCY_BASE   = 0.20
PERFORMANCE_RATIO = 0.8

CV_YEARS         = list(range(2018, 2023))  # use 2018–2022 for CV
TEST_YEAR        = 2023
HOURS_PER_YEAR   = 24 * 365

# === Helper Functions ===
def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    parts = []
    for yr in years:
        p = Path(base_dir) / f"solar_{yr}.csv"
        if not p.exists():
            sys.exit(f"Missing file: {p}")
        df = pd.read_csv(p, skiprows=2)
        df['datetime'] = pd.to_datetime(
            dict(year=df.Year, month=df.Month, day=df.Day,
                 hour=df.Hour, minute=df.Minute)
        )
        for c in ['GHI','DHI','DNI','Temperature','Wind Speed',
                  'Relative Humidity','Pressure','Cloud Type','Dew Point']:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        parts.append(df.set_index('datetime').dropna(subset=['GHI']))
    return pd.concat(parts).sort_index()

def compute_zenith(df):
    phi = np.radians(LATITUDE)
    doy = df.index.dayofyear.values
    delta = np.radians(23.44) * np.sin(2*np.pi*(doy + 284)/365)
    hour = df.index.hour + df.index.minute/60
    H = np.radians((hour - 12) * 15)
    cosZ = np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(H)
    cosZ = np.clip(cosZ, -1, 1)
    return np.degrees(np.arccos(cosZ))

def prepare_features(df):
    df = df.copy()
    df['zenith'] = compute_zenith(df)
    # time and weather features
    df['sin_hour']    = np.sin(2*np.pi * df.index.hour/24)
    df['cos_hour']    = np.cos(2*np.pi * df.index.hour/24)
    df['dayofyear']   = df.index.dayofyear
    df['zenith_norm'] = df['zenith']/90.0
    df['is_clear']    = (df['Cloud Type']==0).astype(int)
    # lagged GHI & rolling stats
    df['ghi_lag_1']       = df['GHI'].shift(1)
    df['ghi_lag_24']      = df['GHI'].shift(24)
    df['ghi_lag_168']     = df['GHI'].shift(168)
    df['ghi_roll24_mean'] = df['GHI'].rolling(24).mean().shift(1)
    df['ghi_roll24_std']  = df['GHI'].rolling(24).std().shift(1)
    # select features
    feat_cols = [
        'sin_hour','cos_hour','dayofyear','zenith_norm','is_clear',
        'DHI','DNI','Temperature','Relative Humidity','Dew Point',
        'ghi_lag_1','ghi_lag_24','ghi_lag_168',
        'ghi_roll24_mean','ghi_roll24_std'
    ]
    df = df.dropna(subset=feat_cols + ['GHI'])
    X = df[feat_cols]
    y = df['GHI']
    return X, y

def ghi_to_power(ghi, zenith):
    perf_corr = np.cos(np.radians(zenith))
    adj_ghi    = ghi * perf_corr
    return (adj_ghi * PANEL_AREA_TOTAL * EFFICIENCY_BASE * PERFORMANCE_RATIO) / 1e6  # MW

# === Main Pipeline ===
def main():
    os.makedirs("model_results", exist_ok=True)

    # 1. Load & feature-engineer
    df = load_solar_data()

    # 1a. Compute solar zenith and attach to the main DataFrame
    df['zenith'] = compute_zenith(df)

    # 1b. Feature‐engineering (lags, rolling stats, time features, etc.)
    X_all, y_all = prepare_features(df)

    # 2. Split CV vs. hold-out
    mask_cv   = X_all.index.year <= 2022
    mask_test = X_all.index.year == TEST_YEAR
    X_cv, y_cv     = X_all.loc[mask_cv], y_all.loc[mask_cv]
    X_test, y_test = X_all.loc[mask_test], y_all.loc[mask_test]

    # 3. Precompute splits
    tscv   = TimeSeriesSplit(n_splits=4, test_size=HOURS_PER_YEAR)
    splits = list(tscv.split(X_cv))

    # 4. Bayesian hyperparameter tuning
    def objective(trial):
        params = {
            'n_estimators':   trial.suggest_int('n_estimators', 100, 500),
            'learning_rate':  trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'minibatch_frac': trial.suggest_uniform('minibatch_frac', 0.1, 1.0),
            'natural_gradient': True
        }
        rmses = []
        for idx, (tr, va) in enumerate(splits, start=1):
            X_tr, y_tr = X_cv.iloc[tr], y_cv.iloc[tr]
            X_va, y_va = X_cv.iloc[va], y_cv.iloc[va]
            model = NGBRegressor(Dist=Normal, Score=MLE, **params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            rmse = mean_squared_error(y_va, preds)
            trial.report(rmse, idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
            rmses.append(rmse)
        return np.mean(rmses)

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    with open('model_results/solar_ngboost_best_params.json','w') as f:
        json.dump(best_params, f, indent=2)

    # 5. CV with best_params
    cv_log = []
    for i, (tr, va) in enumerate(splits, start=1):
        model = NGBRegressor(Dist=Normal, Score=MLE, **best_params)
        model.fit(X_cv.iloc[tr], y_cv.iloc[tr])
        preds = model.predict(X_cv.iloc[va])
        rmse = mean_squared_error(y_cv.iloc[va], preds)
        cv_log.append({'fold': i, 'rmse_Wm2': rmse})
        print(f"Fold {i} RMSE (W/m²): {rmse:.2f}")
    pd.DataFrame(cv_log).to_csv('model_results/solar_ngboost_cv.csv', index=False)

    # 6. Final train & hold-out test
    final = NGBRegressor(Dist=Normal, Score=MLE, **best_params)
    final.fit(X_cv, y_cv)
    y_pred_test = final.predict(X_test)
    rmse_test   = mean_squared_error(y_test, y_pred_test)
    print(f"2023 Hold-out RMSE (W/m²): {rmse_test:.2f}")
    pd.DataFrame([{'year': TEST_YEAR, 'rmse_Wm2': rmse_test}]) \
      .to_csv('model_results/solar_ngboost_holdout_rmse.csv', index=False)

    # 7. Save hold-out forecasts + power
    out = pd.DataFrame({
        'datetime':       X_test.index,
        'GHI_true':       y_test,
        'GHI_pred':       y_pred_test,
        'power_true_MW':  ghi_to_power(y_test.values, df.loc[y_test.index,'zenith']),
        'power_pred_MW':  ghi_to_power(y_pred_test, df.loc[y_test.index,'zenith'])
    })
    out.to_csv('model_results/solar_ngboost_holdout_forecast.csv', index=False)
    print("All NGBoost solar results saved under model_results/")

if __name__ == '__main__':
    main()
