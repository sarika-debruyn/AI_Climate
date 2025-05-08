import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor
import pvlib
import warnings

warnings.filterwarnings("ignore")

# === Constants ===
LATITUDE           = 32.7
LONGITUDE          = -114.63
EFFICIENCY_BASE    = 0.20
TEMP_COEFF         = 0.004
T_REF              = 25.0
TOTAL_FARM_AREA    = 256_000      # m² for 40 MW farm
MAX_SAMPLES        = 10_000
CV_YEARS           = list(range(2018, 2022))  # folds: test years 2019,2020,2021,2022
TEST_YEAR          = 2023
HOURS_PER_YEAR     = 24 * 365

# === Utility Functions ===
def load_solar_data(base_dir="solar_data", years=range(2018, 2024)):
    paths = [Path(base_dir)/f"solar_{y}.csv" for y in years]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, skiprows=2)
        df["datetime"] = pd.to_datetime(
            dict(year=df.Year, month=df.Month, day=df.Day,
                 hour=df.Hour, minute=df.Minute)
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).set_index("datetime").sort_index()
    # ensure numeric
    for col in ["GHI","DHI","DNI","Temperature","Wind Speed",
                "Relative Humidity","Pressure","Cloud Type"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["GHI"])

def add_zenith(df):
    sp = pvlib.solarposition.get_solarposition(
        time=df.index, latitude=LATITUDE, longitude=LONGITUDE
    )
    df["zenith"] = sp["zenith"]
    return df

def prepare_features(df):
    df = df.copy()
    df["sin_h"]   = np.sin(2*np.pi*df.index.hour/24)
    df["cos_h"]   = np.cos(2*np.pi*df.index.hour/24)
    df["doy"]     = df.index.dayofyear
    df["zenith_n"]= df["zenith"]/90.0
    df["is_clear"]= (df["Cloud Type"]==0).astype(int)
    feat_cols = ["sin_h","cos_h","doy","zenith_n",
                 "DHI","DNI","Temperature","Relative Humidity",
                 "Wind Speed","is_clear"]
    X = df[feat_cols].dropna()
    y = df.loc[X.index, "GHI"]
    return X, y

def ghi_to_power(ghi, temp):
    eff = EFFICIENCY_BASE * (1 - TEMP_COEFF*(temp - T_REF))
    return ghi * TOTAL_FARM_AREA * eff / 1e6  # MW

# === Main ===
def main():
    os.makedirs("model_results", exist_ok=True)

    # 1. Load & prepare
    df  = load_solar_data()
    df  = add_zenith(df)
    X_all, y_all = prepare_features(df)

    # 2. Split CV vs. final test
    mask_cv   = X_all.index.year.isin(CV_YEARS + [CV_YEARS[0]])  # for expanding window
    mask_test = X_all.index.year == TEST_YEAR

    X_cv, y_cv     = X_all.loc[X_all.index.year <= 2022], y_all.loc[y_all.index.year <= 2022]
    X_test, y_test = X_all.loc[mask_test], y_all.loc[mask_test]

    # 3. Cross-validation on 2018–2022
    tscv = TimeSeriesSplit(n_splits=len(CV_YEARS), test_size=HOURS_PER_YEAR)
    cv_records = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), start=1):
        X_tr, y_tr = X_cv.iloc[train_idx], y_cv.iloc[train_idx]
        X_va, y_va = X_cv.iloc[val_idx], y_cv.iloc[val_idx]

        # optional subsample
        if len(X_tr) > MAX_SAMPLES:
            sel = np.random.choice(len(X_tr), MAX_SAMPLES, replace=False)
            X_tr, y_tr = X_tr.iloc[sel], y_tr.iloc[sel]

        model = TabPFNRegressor(device=device)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_va)
        rmse   = mean_squared_error(y_va, y_pred)

        cv_records.append({"fold": fold, "rmse_GHI": rmse})
        print(f"Fold {fold} RMSE (GHI): {rmse:.2f}")

    pd.DataFrame(cv_records).to_csv("model_results/solar_tabpfn_cv.csv", index=False)

    # 4. Final train on all 2018–2022, test on 2023
    model_final = TabPFNRegressor(device=device)
    model_final.fit(X_cv, y_cv)

    y_test_pred = model_final.predict(X_test)
    rmse_test   = mean_squared_error(y_test, y_test_pred)
    print(f"\n2023 Hold-out RMSE (GHI): {rmse_test:.2f}")

    # save hold-out results
    pd.DataFrame([{"year": TEST_YEAR, "rmse_GHI": rmse_test}]) \
      .to_csv("model_results/solar_tabpfn_2023_holdout_rmse.csv", index=False)

    # (Optional) convert hold-out GHI preds to power
    power_true = ghi_to_power(y_test.values, X_test["Temperature"].values)
    power_pred = ghi_to_power(y_test_pred,  X_test["Temperature"].values)
    df_out = pd.DataFrame({
        "datetime": X_test.index,
        "GHI_true": y_test,
        "GHI_pred": y_test_pred,
        "power_true_MW": power_true,
        "power_pred_MW": power_pred
    })
    df_out.to_csv("model_results/solar_tabpfn_2023_holdout_forecast.csv", index=False)
    print("Saved 2023 hold-out forecasts to model_results/solar_tabpfn_2023_holdout_forecast.csv")

if __name__ == "__main__":
    main()
