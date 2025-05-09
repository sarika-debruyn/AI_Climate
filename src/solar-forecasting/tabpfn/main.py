#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import json
import warnings

import pandas as pd
import numpy as np
import torch
import pvlib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")

# === Configuration ===
SOLAR_DIR      = "../solar_data"
MODEL_RESULTS  = "../../../model_results/solar"
CV_YEARS       = list(range(2018, 2023))  # train on 2018–2022
TEST_YEAR      = 2023
HOURS_PER_YEAR = 24 * 365

PANEL_AREA     = 256_000    # m²
EFFICIENCY     = 0.20
PERF_RATIO     = 0.8
TEMP_COEFF     = 0.004
T_REF          = 25.0

# === Load Data & Climatology ===
def load_solar_data(base_dir="../../solar_data", years=range(2018, 2024)):
    parts = []
    for yr in years:
        f = Path(base_dir) / f"solar_{yr}.csv"
        if not f.exists():
            sys.exit(f"Missing file: {f}")
        df = pd.read_csv(f, header=0)
        # parse datetime
        df['datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
        df = df.set_index('datetime').drop(columns=['Year','Month','Day','Hour','Minute'])
        # cast dtypes
        df = df.astype({
            'GHI': float,
            'Temperature': float,
            'DHI': float,
            'DNI': float,
            'Relative Humidity': float,
            'Dew Point': float,
            'Cloud Type': int
        }).dropna(subset=['GHI'])
        parts.append(df)
    return pd.concat(parts).sort_index()


def build_climatology(df):
    hist = df[df.index.year.isin(CV_YEARS)]
    clim = hist['GHI'].groupby([hist.index.month, hist.index.hour]).mean()
    return clim

# === Feature Engineering ===
def prepare_features(df, clim):
    df = df.copy()
    # climatology residual
    months = df.index.month
    hours  = df.index.hour
    df['GHI_clim'] = clim.loc[list(zip(months, hours))].values
    df['resid']    = df['GHI'] - df['GHI_clim']
    # time harmonics
    df['sin_h'] = np.sin(2*np.pi * hours/24)
    df['cos_h'] = np.cos(2*np.pi * hours/24)
    doy = df.index.dayofyear
    df['sin_doy'] = np.sin(2*np.pi * doy/365)
    df['cos_doy'] = np.cos(2*np.pi * doy/365)
    # temp derate
    df['temp_diff'] = df['Temperature'] - T_REF
    df['eff_temp'] = 1 - TEMP_COEFF * df['temp_diff']
    # ordinal cloud type
    df['cloud_type'] = df['Cloud Type'].astype(int)
    # lags & rolling stats
    lags = [1,3,6,24,168]
    for var in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
        df[f'{var}_roll24_mean'] = df[var].rolling(24).mean().shift(1)
        df[f'{var}_roll24_std']  = df[var].rolling(24).std().shift(1)
    # drop rows with NaNs
    feat_cols = [
        'sin_h','cos_h','sin_doy','cos_doy',
        'temp_diff','eff_temp','cloud_type'
    ] + [f'{v}_lag{lag}' for v in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point'] for lag in lags] + \
      [f'{v}_roll24_mean' for v in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']] + \
      [f'{v}_roll24_std'  for v in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']]
    df = df.dropna(subset=feat_cols + ['resid'])
    X = df[feat_cols]
    y = df['resid']
    # also return climatology for these rows
    clim_vals = df['GHI_clim'].values
    return X, y, clim_vals

# === Main ===
def main():
    os.makedirs(MODEL_RESULTS, exist_ok=True)
    # load + climatology
    df = load_solar_data()
    clim = build_climatology(df)
    # prepare features
    X_all, y_resid, clim_all = prepare_features(df, clim)
    # split train/test
    train_mask = X_all.index.year.isin(CV_YEARS)
    test_mask  = X_all.index.year == TEST_YEAR
    X_tr, y_tr, clim_tr = X_all.loc[train_mask], y_resid[train_mask], clim_all[train_mask]
    X_te, y_te, clim_te = X_all.loc[test_mask], y_resid[test_mask], clim_all[test_mask]
    # time-series CV
    tscv = TimeSeriesSplit(n_splits=len(CV_YEARS)-1, test_size=HOURS_PER_YEAR)
    splits = list(tscv.split(X_tr))
    # tuning
    def objective(trial):
        params = {
            'n_estimators':   trial.suggest_int('n_estimators',100,500),
            'learning_rate':  trial.suggest_loguniform('learning_rate',1e-3,1e-1),
            'minibatch_frac': trial.suggest_uniform('minibatch_frac',0.1,1.0),
            'natural_gradient': True
        }
        errs = []
        for tr_idx, va_idx in splits:
            m = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p = m.predict(X_tr.iloc[va_idx])
            errs.append(mean_squared_error(y_tr.iloc[va_idx], p))
        return np.mean(errs)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    best = study.best_params
    json.dump(best, open(f'{MODEL_RESULTS}/solar_tabpfn_best.json','w'), indent=2)
    # final train
    model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', **best)
    model.fit(X_tr, y_tr)
    # predict residuals and reconstruct GHI
    resid_pred = model.predict(X_te)
    ghi_pred   = clim_te + resid_pred
    ghi_true   = clim_te + y_te
    # evaluate
    rmse_ghi = mean_squared_error(ghi_true, ghi_pred)
    pd.DataFrame([{'year':TEST_YEAR,'rmse_Wm2':rmse_ghi}]) \
      .to_csv(f'{MODEL_RESULTS}/solar_tabpfn_holdout_rmse.csv',index=False)
    # convert to power via pvlib zenith
    solpos = pvlib.solarposition.get_solarposition(df.loc[test_mask].index, LATITUDE, LONGITUDE)
    zen    = solpos['zenith'].to_numpy(dtype=float)
    perf   = np.cos(zen * np.pi/180)
    p_true = ghi_true * perf * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6
    p_pred = ghi_pred * perf * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6
    pd.DataFrame({
        'datetime':     df.loc[test_mask].index,
        'power_true_MW': p_true,
        'power_pred_MW': p_pred
    }).to_csv(f'{MODEL_RESULTS}/solar_tabpfn_holdout_forecast.csv', index=False)
    print('Done: TabPFN residual pipeline complete')

if __name__ == '__main__':
    main()
