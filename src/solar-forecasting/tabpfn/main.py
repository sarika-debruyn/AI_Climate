
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import os
import warnings
import pandas as pd
import numpy as np
import torch
import pvlib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")

# ─── Centralised paths & helpers ─────────────────────────────────────────
from shared.path_utils import (
    SOLAR_DATA_DIR,
    solar_output,
    _ensure_dirs
)

warnings.filterwarnings("ignore")
_ensure_dirs()  # ensure model_results/solar/outputs exists


# === Configuration ===
LATITUDE        = 32.7
LONGITUDE       = -114.63
MODEL_RESULTS   = "../../../model_results/solar"
CV_YEARS        = list(range(2018, 2023))  # train on 2018–2022
TEST_YEAR       = 2023
HOURS_PER_YEAR  = 24 * 365

PANEL_AREA      = 256_000    # m²
EFFICIENCY      = 0.20
PERF_RATIO      = 0.8
TEMP_COEFF      = 0.004
T_REF           = 25.0

# === Load Data & Climatology ===
def load_solar_data(base_dir=SOLAR_DATA_DIR, years=range(2018, 2024)):
    parts = []
    for yr in years:
        f = Path(base_dir) / f"solar_{yr}.csv"
        if not f.exists():
            sys.exit(f"Missing file: {f}")
        # read header so Year...Minute columns are present
        df = pd.read_csv(f, header=2)
        # build datetime index
        df['datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
        df = df.drop(columns=['Year','Month','Day','Hour','Minute'])
        # cast and drop missing
        df = (
            df.set_index('datetime')
              .astype({
                'GHI': float,
                'Temperature': float,
                'DHI': float,
                'DNI': float,
                'Relative Humidity': float,
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

# === Feature Engineering ===
def prepare_features(df, clim):
    df = df.copy()
    # residual target
    mon, hr = df.index.month, df.index.hour
    df['GHI_clim'] = clim.loc[list(zip(mon, hr))].values
    df['resid']    = df['GHI'] - df['GHI_clim']
    # time features
    hours = hr
    df['sin_h']   = np.sin(2*np.pi * hours/24)
    df['cos_h']   = np.cos(2*np.pi * hours/24)
    doy = df.index.dayofyear
    df['sin_doy'] = np.sin(2*np.pi * doy/365)
    df['cos_doy'] = np.cos(2*np.pi * doy/365)
    # temperature derate
    df['temp_diff'] = df['Temperature'] - T_REF
    df['eff_temp']  = 1 - TEMP_COEFF * df['temp_diff']
    # ordinal cloud type
    df['cloud_type'] = df['Cloud Type'].astype(int)
    # lags & rolling stats
    lags = [1,3,6,24,168]
    for var in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
        df[f'{var}_roll24_mean'] = df[var].rolling(24).mean().shift(1)
        df[f'{var}_roll24_std']  = df[var].rolling(24).std().shift(1)
    # drop rows with missing
    feat_cols = [
        'sin_h','cos_h','sin_doy','cos_doy',
        'temp_diff','eff_temp','cloud_type'
    ] + [f'{v}_lag{lag}' for v in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point'] for lag in lags] + \
      [f'{v}_roll24_mean' for v in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']] + \
      [f'{v}_roll24_std'  for v in ['GHI','DHI','DNI','Temperature','Relative Humidity','Dew Point']]
    df = df.dropna(subset=feat_cols + ['resid'])
    X = df[feat_cols]
    y = df['resid']
    clim_vals = df['GHI_clim'].values
    return X, y, clim_vals

# === Main ===

def main():
    os.makedirs(MODEL_RESULTS, exist_ok=True)

    # 1) load & baseline
    df   = load_solar_data()
    clim = build_climatology(df)

    # 2) featurize & get residual target + hourly climatology array
    X_all, y_resid_all, clim_all = prepare_features(df, clim)

    # 3) split into CV / test
    train_mask = X_all.index.year.isin(CV_YEARS)
    test_mask  = X_all.index.year == TEST_YEAR

    X_tr, y_tr      = X_all.loc[train_mask], y_resid_all[train_mask]
    X_te, y_te      = X_all.loc[test_mask],   y_resid_all[test_mask]
    # reconstruct true GHI for test
    ghi_te_full     = clim_all[test_mask] + y_te
    test_times      = X_te.index

    # 4) CV residual‐training
    tscv    = TimeSeriesSplit(n_splits=len(CV_YEARS)-1, test_size=HOURS_PER_YEAR)
    cv_log  = []
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_tr), 1):
        m = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        resid_pred_va = m.predict(X_tr.iloc[va_idx])
        rmse = np.sqrt(mean_squared_error(y_tr.iloc[va_idx], resid_pred_va))
        cv_log.append({"fold": fold, "rmse_resid_Wm2": rmse})
        print(f"Fold {fold} RMSE (resid): {rmse:.2f}")
    
    pd.DataFrame(cv_log).to_csv(
        solar_output("solar_tabpfn_cv.csv"),
        index=False
    )
    # 5) final train & predict residuals on test
    final = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
    final.fit(X_tr, y_tr)
    resid_pred_te = final.predict(X_te)

    # 6) reconstruct full GHI forecast
    ghi_pred_full = clim_all[test_mask] + resid_pred_te

    # 7) compute RMSE on full GHI
    rmse_ghi = np.sqrt(mean_squared_error(ghi_te_full, ghi_pred_full))
    print(f"2023 Hold-out RMSE (GHI): {rmse_ghi:.2f}")
    pd.DataFrame([{"year": TEST_YEAR, "rmse_Wm2": rmse_ghi}]) \
      .to_csv(solar_output("solar_tabpfn_holdout_rmse.csv"), index=False)

    # 8) convert both true & forecast GHI → Power (MW)
    solpos = pvlib.solarposition.get_solarposition(test_times, LATITUDE, LONGITUDE)
    zen    = solpos["zenith"].to_numpy(dtype=float)
    cosz   = np.cos(zen * np.pi/180)

    p_true = ghi_te_full * cosz * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6
    p_pred = ghi_pred_full * cosz * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6

    # 9) clip negative → 0
    p_true = np.maximum(0, p_true)
    p_pred = np.maximum(0, p_pred)

    # 10) save hourly forecasts
    pd.DataFrame({
        "datetime":      times_te,
        "power_true_MW": p_true,
        "power_pred_MW": p_pred
    }).to_csv(
        solar_output("solar_tabpfn_holdout_forecast.csv"),
        index=False
    )

    print("Done — full-GHI & power forecasts saved.")

if __name__ == "__main__":
    main()
