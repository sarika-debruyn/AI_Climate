# File: src/solar-forecasting/NGBoost/main.py
#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))   # add …/src
import json
import warnings
import numpy as np
import optuna
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ── Centralised repo paths ──────────────────────────────────────────────
from shared.path_utils import SOLAR_DATA_DIR, solar_output

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────
CV_YEARS        = list(range(2018, 2023))   # 2018–2022
TEST_YEAR       = 2023
HOURS_PER_YEAR  = 24 * 365

# Solar‑farm parameters
PANEL_AREA      = 256_000    # m²  (≈40 MW farm)
EFFICIENCY      = 0.20       # nominal module efficiency
PERF_RATIO      = 0.8        # whole‑system performance ratio
TEMP_COEFF      = 0.004      # efficiency loss / °C above reference
T_REF           = 25.0       # reference module temperature (°C)

# ────────────────────────────────────────────────────────────────────────
# Data loading & climatology
# ────────────────────────────────────────────────────────────────────────
def load_solar_data(base_dir = SOLAR_DATA_DIR, years=range(2018, 2024)):
    parts = []
    for yr in years:
        p = base_dir / f"solar_{yr}.csv"
        if not p.exists():
            sys.exit(f"[ERROR] Missing file: {p}")
        df = pd.read_csv(p, header=2)

        # Build timestamp index
        df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
        df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"]).set_index("datetime")

        # Ensure numeric types & drop rows without GHI
        numeric_cols = [
            "GHI", "Temperature", "DHI", "DNI", "Wind Speed",
            "Relative Humidity", "Pressure", "Ozone", "Dew Point"
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["Cloud Type"] = df["Cloud Type"].astype(int)
        df = df.dropna(subset=["GHI"]).sort_index()

        parts.append(df)

    return pd.concat(parts)

def build_climatology(df: pd.DataFrame) -> pd.Series:
    hist = df[df.index.year.isin(CV_YEARS)]
    return hist["GHI"].groupby([hist.index.month, hist.index.hour]).mean()

# ────────────────────────────────────────────────────────────────────────
# Feature engineering (residual learning)
# ────────────────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame, clim: pd.Series):
    df = df.copy()

    # Climatology residual
    mon, hr = df.index.month, df.index.hour
    df["GHI_clim"] = clim.loc[list(zip(mon, hr))].values
    df["resid"]    = df["GHI"] - df["GHI_clim"]

    # Cyclic time features
    df["sin_h"]  = np.sin(2 * np.pi * hr / 24)
    df["cos_h"]  = np.cos(2 * np.pi * hr / 24)
    doy          = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    # Temperature‑derivative efficiency terms
    df["temp_diff"] = df["Temperature"] - T_REF
    df["eff_temp"]  = 1 - TEMP_COEFF * df["temp_diff"]

    # Cloud‑type (categorical converted to integer already)
    df["cloud_type"] = df["Cloud Type"]

    # Lags & rolling stats
    lags = [1, 3, 6, 24, 168]  # hours
    base_vars = [
        "GHI", "Temperature", "DHI", "DNI", "Wind Speed",
        "Relative Humidity", "Ozone", "Dew Point"
    ]
    for var in base_vars:
        for lag in lags:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
        df[f"{var}_roll24_mean"] = df[var].rolling(24).mean().shift(1)
        df[f"{var}_roll24_std"]  = df[var].rolling(24).std().shift(1)

    # Final feature list
    feat_cols  = ["sin_h", "cos_h", "sin_doy", "cos_doy",
                  "temp_diff", "eff_temp", "cloud_type"]
    feat_cols += [f"{v}_lag{l}" for v in base_vars for l in lags]
    feat_cols += [f"{v}_roll24_{stat}" for v in base_vars for stat in ("mean", "std")]

    # Drop rows with NA (from lag/rolling creation)
    df = df.dropna(subset=feat_cols + ["resid"])

    X        = df[feat_cols]
    y_resid  = df["resid"]
    ghi_true = df["GHI"]
    return X, y_resid, ghi_true

# ────────────────────────────────────────────────────────────────────────
# Main training / evaluation routine
# ────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data & build features
    df   = load_solar_data()
    clim = build_climatology(df)
    X, y_resid, ghi_true = prepare_features(df, clim)

    # 2. Train/test split
    train_mask = X.index.year.isin(CV_YEARS)
    test_mask  = X.index.year == TEST_YEAR
    X_tr, y_tr = X.loc[train_mask], y_resid.loc[train_mask]
    X_te, y_te = X.loc[test_mask],  y_resid.loc[test_mask]
    ghi_te     = ghi_true.loc[test_mask]

    # 3. Time‑series CV splits (one fold per year after the first)
    tscv   = TimeSeriesSplit(n_splits=len(CV_YEARS) - 1, test_size=HOURS_PER_YEAR)
    splits = list(tscv.split(X_tr))

    # 4. Hyperparameter search with Optuna
    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 100, 400),
            "learning_rate":  trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "minibatch_frac": trial.suggest_uniform("minibatch_frac", 0.1, 0.8),
            "natural_gradient": True,
        }
        rmses = []
        for tr_idx, va_idx in splits:
            model = NGBRegressor(Dist=Normal, Score=MLE, **params)
            model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pred = model.predict(X_tr.iloc[va_idx])
            rmses.append(np.sqrt(mean_squared_error(y_tr.iloc[va_idx], pred)))
        return np.mean(rmses)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    study.optimize(objective, n_trials=15)
    best_params = study.best_params

    # Save best params
    with open(solar_output("solar_ngboost_best.json"), "w") as fp:
        json.dump(best_params, fp, indent=2)

    # 5. Train final model on full training data
    model = NGBRegressor(Dist=Normal, Score=MLE, **best_params)
    model.fit(X_tr, y_tr)

    # 6. Predict 2023 residuals & reconstruct GHI
    resid_pred = model.predict(X_te)
    ghi_clim   = clim.loc[list(zip(X_te.index.month, X_te.index.hour))].values
    ghi_pred   = ghi_clim + resid_pred

    # Hold‑out RMSE
    rmse_test = np.sqrt(mean_squared_error(ghi_te, ghi_pred))
    print(f"2023 Hold‑out RMSE: {rmse_test:.2f} W m⁻²")

    pd.DataFrame([{"year": TEST_YEAR, "rmse_Wm2": rmse_test}]).to_csv(
        solar_output("solar_ngboost_holdout_rmse.csv"), index=False
    )

    # 7. Convert GHI → AC power (very simplified cosine‑zenith model)
    hr         = X_te.index.hour.values
    zen_angle  = np.abs(12 - hr) / 12 * 90          # zenith in degrees
    perf_ratio = np.cos(np.radians(zen_angle))      # crude cosine loss
    p_true_MW  = ghi_te.values * perf_ratio * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6
    p_pred_MW  = ghi_pred     * perf_ratio * PANEL_AREA * EFFICIENCY * PERF_RATIO / 1e6

    pd.DataFrame({
        "datetime":      X_te.index,
        "power_true_MW": p_true_MW,
        "power_pred_MW": p_pred_MW,
    }).to_csv(
        solar_output("solar_ngboost_holdout_forecast.csv"), index=False
    )

    # Save results
    output_dir = Path("../model_results/solar/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual model results
    results = {"NGBoost": pd.DataFrame({"datetime": X_te.index, "power_pred_MW": p_pred_MW})}
    for model_name, forecasts in results.items():
        out_path = output_dir / f"{model_name}_forecasts.csv"
        forecasts.to_csv(out_path)
        print(f"Saved {model_name} forecasts to {out_path}")
    
    # Merge all forecasts
    merged_forecasts = pd.concat(results.values(), axis=1)
    merged_forecasts.columns = results.keys()
    
    # Save merged forecasts
    merged_path = output_dir / "solar_merged_forecasts.csv"
    merged_forecasts.to_csv(merged_path)
    print(f"Saved merged forecasts to {merged_path}")

    print(" NGBoost residual pipeline complete. Outputs saved to model_results/solar/outputs/")

# ── Entry‑point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
