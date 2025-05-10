from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))   # add …/src
import warnings
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor

from shared.path_utils import WIND_DATA_DIR, wind_output, _ensure_dirs

warnings.filterwarnings("ignore")
_ensure_dirs()  # ensure model_results/wind/outputs exists

# === Constants & Parameters ===
AIR_DENSITY    = 1.121    # kg/m³
TURBINE_RADIUS = 50       # m
SWEEP_AREA     = np.pi * TURBINE_RADIUS**2
EFFICIENCY     = 0.40
TURBINE_COUNT  = 16
MAX_SAMPLES    = 10_000

# Cross-validation on 2018–2022, test on 2023
CV_YEARS       = list(range(2018, 2023))  # [2018,2019,2020,2021,2022]
TEST_YEAR      = 2023
HOURS_PER_YEAR = 24 * 365

# === Helper Functions ===
def wind_to_power(ws):
    coef = 0.5 * AIR_DENSITY * SWEEP_AREA * EFFICIENCY * TURBINE_COUNT
    return coef * (ws ** 3) / 1000  # kW

def load_wind_data(base_dir=WIND_DATA_DIR):
    parts = []
    for year in range(2018, TEST_YEAR + 1):
        path = Path(base_dir) / f"wind_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = (pd.read_csv(path, skiprows=2)
                .assign(datetime=lambda d: pd.to_datetime(
                    dict(year=d.Year, month=d.Month, day=d.Day,
                         hour=d.Hour, minute=d.Minute)
                ))
                .set_index("datetime"))
        parts.append(df[['Wind Speed','Temperature','Relative Humidity',
                         'Pressure','Cloud Type','Dew Point']].apply(pd.to_numeric, errors='coerce'))
    df = pd.concat(parts).sort_index().dropna(subset=['Wind Speed'])
    return df

def make_features(df):
    df = df.copy()
    df['power_kW'] = wind_to_power(df['Wind Speed'])
    # time features
    df['sin_h']    = np.sin(2 * np.pi * df.index.hour / 24)
    df['cos_h']    = np.cos(2 * np.pi * df.index.hour / 24)
    df['doy']      = df.index.dayofyear
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)
    # lags & rolling stats
    for lag in (1, 24, 168):
        df[f'lag_{lag}'] = df['power_kW'].shift(lag)
    df['roll24_mean'] = df['power_kW'].rolling(24).mean().shift(1)
    df['roll24_std']  = df['power_kW'].rolling(24).std().shift(1)
    # drop NaNs
    feat_cols = [
        'Wind Speed','Temperature','Relative Humidity','Pressure','Dew Point',
        'sin_h','cos_h','doy','is_clear',
        'lag_1','lag_24','lag_168','roll24_mean','roll24_std'
    ]
    df = df.dropna(subset=feat_cols + ['power_kW'])
    return df[feat_cols], df['power_kW']

# === Main Pipeline ===
def main():
    # 1. Load & feature-engineer
    df = load_wind_data()
    X_all, y_all = make_features(df)

    # 2. Split CV vs. final test
    mask_cv   = X_all.index.year.isin(CV_YEARS)
    mask_test = X_all.index.year == TEST_YEAR

    X_cv, y_cv     = X_all.loc[mask_cv], y_all.loc[mask_cv]
    X_test, y_test = X_all.loc[mask_test], y_all.loc[mask_test]

    # 3. Cross-validation (2018–2022)
    tscv = TimeSeriesSplit(n_splits=len(CV_YEARS)-1, test_size=HOURS_PER_YEAR)
    cv_records = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), start=1):
        X_tr, y_tr = X_cv.iloc[train_idx], y_cv.iloc[train_idx]
        X_va, y_va = X_cv.iloc[val_idx], y_cv.iloc[val_idx]

        # subsample if dataset is large
        if len(X_tr) > MAX_SAMPLES:
            sel = np.random.choice(len(X_tr), MAX_SAMPLES, replace=False)
            X_tr, y_tr = X_tr.iloc[sel], y_tr.iloc[sel]

        # train & predict
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)

        # record RMSE
        rmse = mean_squared_error(y_va, y_pred)
        print(f"Fold {fold} RMSE (kW): {rmse:.2f}")
        cv_records.append({'fold': fold, 'rmse_kW': rmse})

    pd.DataFrame(cv_records).to_csv(
        wind_output("wind_tabpfn_cv.csv"), index=False
    )

    # 4. Final train on 2018–2022, test on 2023
    model_final = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
    model_final.fit(X_cv, y_cv)

    y_test_pred = model_final.predict(X_test)
    rmse_test   = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"\n2023 Hold-out RMSE (kW): {rmse_test:.2f}")

    # save hold-out RMSE

    pd.DataFrame([{'year': TEST_YEAR, 'rmse_kW': rmse_test}]).to_csv(
        wind_output("wind_tabpfn_2023_holdout_rmse.csv"), index=False
    )

    # save forecasts for downstream analysis
    out = pd.DataFrame({
        'datetime': X_test.index,
        'power_true_kW': y_test,
        'power_pred_kW': y_test_pred
    })
    out.to_csv(
        wind_output("wind_tabpfn_2023_holdout_forecast.csv")
    )
    print("Saved 2023 hold-out forecasts to " + wind_output("wind_tabpfn_2023_holdout_forecast.csv"))

if __name__ == "__main__":
    main()
