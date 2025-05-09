#!/usr/bin/env python3
import os, sys, json, warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
import optuna

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SOLAR_DIR       = "../solar_data"
MODEL_RESULTS   = "../../../model_results/solar"
CV_YEARS        = list(range(2018, 2023))  # train years
TEST_YEAR       = 2023
HOURS_PER_YEAR  = 24 * 365

PANEL_AREA      = 256_000    # m²
EFFICIENCY      = 0.20
PERF_RATIO      = 0.8
TEMP_COEFF      = 0.004
T_REF           = 25.0

# ─────────────────────────────────────────────────────────────
#  LOAD + CLIMATOLOGY
# ─────────────────────────────────────────────────────────────
def load_solar_data(base_dir=SOLAR_DIR, years=range(2018, 2024)):
    parts = []
    for yr in years:
        p = Path(base_dir) / f"solar_{yr}.csv"
        if not p.exists():
            sys.exit(f"Missing file: {p}")
        df = pd.read_csv(p, header=2)
        df['datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
        df = df.drop(columns=['Year','Month','Day','Hour','Minute'])
        df = (
            df.set_index('datetime')
              .astype({
                  'GHI': float,
                  'Temperature': float,
                  'DHI': float,
                  'DNI': float,
                  'Wind Speed': float,
                  'Relative Humidity': float,
                  'Pressure': float,
                  'Ozone': float,
                  'Dew Point': float,
                  'Cloud Type': int
              })
              .sort_index()
              .dropna(subset=['GHI'])
        )
        parts.append(df)
    return pd.concat(parts)

def build_climatology(df):
    hist = df[df.index.year.isin(CV_YEARS)]
    return hist['GHI'].groupby([hist.index.month, hist.index.hour]).mean()

# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING (RESIDUAL LEARNING)
# ─────────────────────────────────────────────────────────────
def prepare_features(df, clim):
    df = df.copy()
    mon, hr = df.index.month, df.index.hour
    df['GHI_clim'] = clim.loc[list(zip(mon, hr))].values
    df['resid']    = df['GHI'] - df['GHI_clim']
    df['sin_h']    = np.sin(2*np.pi * hr/24)
    df['cos_h']    = np.cos(2*np.pi * hr/24)
    doy = df.index.dayofyear
    df['sin_doy']  = np.sin(2*np.pi * doy/365)
    df['cos_doy']  = np.cos(2*np.pi * doy/365)
    df['temp_diff']= df['Temperature'] - T_REF
    df['eff_temp'] = 1 - TEMP_COEFF * df['temp_diff']
    df['cloud_type'] = df['Cloud Type'].astype(int)
    lags = [1,3,6,24,168]
    for var in ['GHI','Temperature','DHI','DNI','Wind Speed','Relative Humidity','Ozone','Dew Point']:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
        df[f'{var}_roll24_mean'] = df[var].rolling(24).mean().shift(1)
        df[f'{var}_roll24_std']  = df[var].rolling(24).std().shift(1)
    feat = ['sin_h','cos_h','sin_doy','cos_doy','temp_diff','eff_temp','cloud_type']
    feat += [f'{v}_lag{l}' for v in ['GHI','Temperature','DHI','DNI','Wind Speed','Relative Humidity','Ozone','Dew Point'] for l in lags]
    feat += [f'{v}_roll24_{stat}' for v in ['GHI','Temperature','DHI','DNI','Wind Speed','Relative Humidity','Ozone','Dew Point'] for stat in ['mean','std']]
    df = df.dropna(subset=feat + ['resid'])
    return df[feat], df['resid'], df['GHI']

# ─────────────────────────────────────────────────────────────
#  MAIN TRAIN / EVAL
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_RESULTS, exist_ok=True)

    df   = load_solar_data()
    clim = build_climatology(df)
    X, y_resid, ghi_true = prepare_features(df, clim)

    train_mask = X.index.year.isin(CV_YEARS)
    test_mask  = X.index.year == TEST_YEAR
    X_tr, y_tr = X.loc[train_mask], y_resid.loc[train_mask]
    X_te, y_te = X.loc[test_mask],  y_resid.loc[test_mask]
    ghi_te     = ghi_true.loc[test_mask]

    tscv   = TimeSeriesSplit(n_splits=len(CV_YEARS)-1, test_size=HOURS_PER_YEAR)
    splits = list(tscv.split(X_tr))

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 100, 400),
            "learning_rate":  trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "minibatch_frac": trial.suggest_uniform("minibatch_frac", 0.1, 0.8),
            "natural_gradient": True
        }
        rmses = []
        for tr_idx, va_idx in splits:
            m = NGBRegressor(Dist=Normal, Score=MLE, **params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p = m.predict(X_tr.iloc[va_idx])
            # now compute RMSE, not MSE
            mse  = mean_squared_error(y_tr.iloc[va_idx], p)   # default returns MSE
            rmse = np.sqrt(mse)
            rmses.append(rmse)
        return np.mean(rmses)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study.optimize(objective, n_trials=15)
    best = study.best_params
    json.dump(best, open(f"{MODEL_RESULTS}/solar/solar_ngboost_best.json","w"), indent=2)

    model = NGBRegressor(Dist=Normal, Score=MLE, **best)
    model.fit(X_tr, y_tr)
    resid_pred = model.predict(X_te)
    ghi_pred   = clim.loc[list(zip(X_te.index.month, X_te.index.hour))].values + resid_pred

    # hold-out RMSE
    mse_test = mean_squared_error(ghi_te, ghi_pred)
    rmse_test = np.sqrt(mse_test)
    print(f"2023 Hold‐out RMSE: {rmse_test:.2f}")
    pd.DataFrame([{"year":TEST_YEAR, "rmse_Wm2":rmse_test}]) \
      .to_csv(f"{MODEL_RESULTS}/solar/solar_ngboost_holdout_rmse.csv", index=False)

    # to power
    hr   = X_te.index.hour.values
    zen  = np.abs(12-hr)/12 * 90
    perf = np.cos(zen * np.pi/180)
    p_true = ghi_te.values * perf * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6
    p_pred = ghi_pred       * perf * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6

    pd.DataFrame({
        "datetime":      X_te.index,
        "power_true_MW": p_true,
        "power_pred_MW": p_pred
    }).to_csv(f"{MODEL_RESULTS}/solar/solar_ngboost_holdout_forecast.csv", index=False)

    print("Done — NGBoost residual pipeline complete.")

if __name__ == "__main__":
    main()
