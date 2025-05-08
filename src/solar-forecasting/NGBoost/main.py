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
import pvlib

warnings.filterwarnings('ignore')

# === Constants & Parameters ===
LATITUDE          = 32.7
LONGITUDE         = -114.63
PANEL_AREA_TOTAL  = 256_000     # m² for 40 MW solar farm
EFFICIENCY_BASE   = 0.20
PERFORMANCE_RATIO = 0.8

CV_YEARS         = list(range(2018, 2023))  # CV on 2018–22
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
        df['datetime'] = pd.to_datetime(dict(
            year=df.Year, month=df.Month, day=df.Day,
            hour=df.Hour, minute=df.Minute
        ))
        for c in ['GHI','DHI','DNI','Temperature','Relative Humidity','Pressure','Cloud Type','Dew Point']:
            df[c] = pd.to_numeric(df.get(c, np.nan), errors='coerce')
        df = df.set_index('datetime').dropna(subset=['GHI'])
        parts.append(df)
    return pd.concat(parts).sort_index()

def compute_zenith(df):
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index, latitude=LATITUDE, longitude=LONGITUDE
    )
    # take the zenith column, drop its index, coerce to float64
    return solpos['zenith'].to_numpy(dtype=float)


def prepare_features(df):
    df = df.copy()
    
    # time features
    df['hour']    = df.index.hour
    df['sin_h1']  = np.sin(2*np.pi * df.hour/24)
    df['cos_h1']  = np.cos(2*np.pi * df.hour/24)
    df['doy']     = df.index.dayofyear
    df['sin_doy'] = np.sin(2*np.pi * df.doy/365)
    df['cos_doy'] = np.cos(2*np.pi * df.doy/365)
    
    # Get solar position
    solpos = pvlib.solarposition.get_solarposition(df.index, LATITUDE, LONGITUDE)
    
    # Clear sky and clearness
    cs = pvlib.clearsky.ineichen(
        solpos['apparent_zenith'], 
        solpos['apparent_elevation'],
        linke_turbidity=3.0
    )
    df['GHI_cs'] = cs['ghi']
    df['k_t']    = df['GHI'] / (df['GHI_cs'] + 1e-6)
    
    # temperature derate
    df['temp_diff'] = df['Temperature'] - 25
    df['eff_temp']  = 1 - 0.004 * df['temp_diff']
    
    # cloud one-hot
    for ct in sorted(df['Cloud Type'].unique()):
        df[f'cloud_{int(ct)}'] = (df['Cloud Type']==ct).astype(int)
    
    # lags & rolling stats
    lags = [1,3,6,24,168]
    for var in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
        df[f'{var}_roll24_mean'] = df[var].rolling(24).mean().shift(1)
        df[f'{var}_roll24_std']  = df[var].rolling(24).std().shift(1)
    
    # interaction
    df['zenith_k_t'] = (solpos['apparent_zenith']/90.0) * df['k_t']
    
    # feature list
    feat_cols = [
        'sin_h1','cos_h1','sin_doy','cos_doy','k_t','eff_temp','zenith_k_t'
    ] \
    + [f'cloud_{int(ct)}' for ct in sorted(df['Cloud Type'].unique())] \
    + [f'{var}_lag{lag}' for var in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point'] for lag in lags] \
    + [f'{var}_roll24_mean' for var in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']] \
    + [f'{var}_roll24_std'  for var in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']]
    
    df = df.dropna(subset=feat_cols + ['GHI'])
    X = df[feat_cols]
    y = df['GHI']
    return X, y

def ghi_to_power(ghi, zenith):
    # Force both inputs into float64 NumPy arrays
    ghi_arr    = np.asarray(ghi, dtype=np.float64)
    zenith_arr = np.asarray(zenith, dtype=np.float64)
    # Compute cosine of zenith in radians without calling np.radians
    perf_corr  = np.cos(zenith_arr * (np.pi / 180.0))
    return (ghi_arr * perf_corr
            * PANEL_AREA_TOTAL
            * EFFICIENCY_BASE
            * PERFORMANCE_RATIO) / 1e6  # MW

# === Main Pipeline ===
def main():
    os.makedirs("../../../model_results", exist_ok=True)

    # load & features
    df = load_solar_data()
    X_all, y_all = prepare_features(df)

    # train/test split
    mask_cv   = X_all.index.year.isin(CV_YEARS)
    mask_test = X_all.index.year == TEST_YEAR
    X_cv, y_cv     = X_all.loc[mask_cv], y_all.loc[mask_cv]
    X_test, y_test = X_all.loc[mask_test], y_all.loc[mask_test]

    # CV splits
    tscv   = TimeSeriesSplit(n_splits=4, test_size=HOURS_PER_YEAR)
    splits = list(tscv.split(X_cv))

    # hyperparameter tuning
    def objective(trial):
        params = {
            'n_estimators':   trial.suggest_int('n_estimators', 200, 600),
            'learning_rate':  trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'minibatch_frac': trial.suggest_uniform('minibatch_frac', 0.2, 0.8),
            'natural_gradient': True
        }
        rmses = []
        for tr, va in splits:
            mdl = NGBRegressor(Dist=Normal, Score=MLE, **params)
            mdl.fit(X_cv.iloc[tr], y_cv.iloc[tr])
            p = mdl.predict(X_cv.iloc[va])
            rmses.append(np.sqrt(mean_squared_error(y_cv.iloc[va], p)))
            trial.report(rmses[-1], len(rmses)-1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(rmses)

    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(objective, n_trials=15)
    best = study.best_params
    json.dump(best, open('../../../model_results/solar_ngboost_best_params.json','w'), indent=2)

    # CV with best params
    cvs = []
    for i, (tr, va) in enumerate(splits,1):
        mdl = NGBRegressor(Dist=Normal, Score=MLE, **best)
        mdl.fit(X_cv.iloc[tr], y_cv.iloc[tr])
        p = mdl.predict(X_cv.iloc[va])
        cvs.append({'fold':i,'rmse_Wm2':np.sqrt(mean_squared_error(y_cv.iloc[va],p))})
    pd.DataFrame(cvs).to_csv('../../../model_results/solar_ngboost_cv.csv',index=False)

    # final train & test
    final = NGBRegressor(Dist=Normal, Score=MLE, **best)
    final.fit(X_cv, y_cv)
    pred = final.predict(X_test)
    rm = np.sqrt(mean_squared_error(y_test, pred))
    pd.DataFrame([{'year':TEST_YEAR,'rmse_Wm2':rm}])\
      .to_csv('../../../model_results/solar_ngboost_holdout_rmse.csv',index=False)

    # save forecasts
    out = pd.DataFrame({
        'datetime':      X_test.index,
        'GHI_true':      y_test,
        'GHI_pred':      pred,
        'power_true_MW': ghi_to_power(y_test.values, df.loc[y_test.index,'zenith']),
        'power_pred_MW': ghi_to_power(pred,      df.loc[y_test.index,'zenith'])
    })
    out.to_csv('../../../model_results/solar_ngboost_holdout_forecast.csv',index=False)
    print("Done ✓ results in model_results/")

if __name__ == '__main__':
    main()