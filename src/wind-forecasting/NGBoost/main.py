# File: src/wind-forecasting/NGBoost/main.py
#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))   # add …/src
import warnings
import json
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE

from shared.path_utils import WIND_DATA_DIR, wind_output, _ensure_dirs

warnings.filterwarnings("ignore")
_ensure_dirs()  # ensure wind output folders exist

# === Constants & Parameters ===
AIR_DENSITY    = 1.121    # kg/m³
TURBINE_RADIUS = 50       # m
SWEEP_AREA     = np.pi * TURBINE_RADIUS**2
EFFICIENCY     = 0.40
TURBINE_COUNT  = 16

CV_YEARS       = list(range(2018, 2023))  # use 2018–2022 for CV
TEST_YEAR      = 2023
HOURS_PER_YEAR = 24 * 365

# === Helper Functions ===
def load_wind_data(base_dir=WIND_DATA_DIR, years=range(2018, 2024)):
    parts = []
    for yr in years:
        path = Path(base_dir) / f"wind_{yr}.csv"
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        df = (pd.read_csv(path, skiprows=2)
                .assign(datetime=lambda d: pd.to_datetime(
                    dict(year=d.Year, month=d.Month, day=d.Day,
                         hour=d.Hour, minute=d.Minute)
                ))
                .set_index('datetime'))
        for col in ['Wind Speed','Temperature','Relative Humidity','Pressure','Cloud Type','Dew Point']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        parts.append(df.dropna(subset=['Wind Speed']))
    return pd.concat(parts).sort_index()

def prepare_features(df):
    df = df.copy()
    # time features
    df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofyear'] = df.index.dayofyear
    df['is_clear']  = (df['Cloud Type'] == 0).astype(int)
    # meteorological covariates
    feat_cols = [
        'sin_hour','cos_hour','dayofyear','is_clear',
        'Temperature','Pressure','Relative Humidity','Dew Point'
    ]
    df = df.dropna(subset=feat_cols + ['Wind Speed'])
    X = df[feat_cols]
    y = df['Wind Speed']
    return X, y

def wind_speed_to_power(ws):
    coef = 0.5 * AIR_DENSITY * SWEEP_AREA * EFFICIENCY * TURBINE_COUNT
    return coef * (ws ** 3) / 1e6  # MW

# === Main ===
def main():
    
    # 1. Load & feature-engineer
    df = load_wind_data()
    X_all, y_all = prepare_features(df)

    # 2. Split CV vs. hold-out
    mask_cv   = X_all.index.year <= 2022
    mask_test = X_all.index.year == TEST_YEAR
    X_cv, y_cv     = X_all.loc[mask_cv], y_all.loc[mask_cv]
    X_test, y_test = X_all.loc[mask_test], y_all.loc[mask_test]

    # 3. Precompute splits (expanding window: test years 2019–2022)
    tscv   = TimeSeriesSplit(n_splits=4, test_size=HOURS_PER_YEAR)
    splits = list(tscv.split(X_cv))

    # 4. Bayesian hyperparameter tuning (10 trials + MedianPruner)
    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
            'learning_rate':   trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'minibatch_frac':  trial.suggest_uniform('minibatch_frac', 0.1, 1.0),
            'natural_gradient': True
        }
        rmses = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr, y_tr = X_cv.iloc[train_idx], y_cv.iloc[train_idx]
            X_va, y_va = X_cv.iloc[val_idx], y_cv.iloc[val_idx]
            model = NGBRegressor(Dist=Normal, Score=MLE, **params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            rmse = mean_squared_error(y_va, y_pred)
            trial.report(rmse, fold_idx)
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
    with open(wind_output('wind_ngboost_best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    # 5. CV with best_params
    cv_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X_cv.iloc[train_idx], y_cv.iloc[train_idx]
        X_va, y_va = X_cv.iloc[val_idx], y_cv.iloc[val_idx]
        model = NGBRegressor(Dist=Normal, Score=MLE, **best_params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        rmse = mean_squared_error(y_va, y_pred)
        cv_records.append({'fold': fold_idx, 'rmse_m_s': rmse})
        print(f"Fold {fold_idx} RMSE (m/s): {rmse:.3f}")
    
    pd.DataFrame(cv_records).to_csv(
        wind_output('wind_ngboost_cv.csv'), index=False
    )

    # 6. Final train on 2018–2022 & hold-out test on 2023
    final_model = NGBRegressor(Dist=Normal, Score=MLE, **best_params)
    final_model.fit(X_cv, y_cv)
    y_test_pred = final_model.predict(X_test)
    rmse_test   = mean_squared_error(y_test, y_test_pred)
    print(f"2023 Hold-out RMSE (m/s): {rmse_test:.3f}")
    pd.DataFrame([{"year": TEST_YEAR, "rmse_m_s": rmse_test}]).to_csv(
        wind_output("wind_ngboost_holdout_rmse.csv"), index=False
    )


    # 7. Save hold-out forecasts and power
    df_out = pd.DataFrame({
        'datetime':       X_test.index,
        'speed_true_m_s': y_test,
        'speed_pred_m_s': y_test_pred
    })
    df_out['power_true_MW'] = wind_speed_to_power(df_out['speed_true_m_s'])
    df_out['power_pred_MW'] = wind_speed_to_power(df_out['speed_pred_m_s'])
    df_out.to_csv(wind_output("wind_ngboost_holdout_forecast.csv"))
    print("Wind NGBoost pipeline complete. Outputs in model_results/wind/outputs/")

    # Save results
    output_dir = Path("../model_results/wind/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual model results
    results = {"NGBoost": pd.DataFrame({"datetime": X_test.index, "power_pred_MW": df_out['power_pred_MW']})}
    
    # Save individual model results
    for model_name, forecasts in results.items():
        out_path = output_dir / f"{model_name}_forecasts.csv"
        forecasts.to_csv(out_path, index=False)
        print(f"Saved {model_name} forecasts to {out_path}")
    
    # Just use the first forecast since there's only one model
    merged_forecasts = results["NGBoost"].copy()
    
    # Save merged forecasts
    merged_path = output_dir / "wind_merged_forecasts.csv"
    merged_forecasts.to_csv(merged_path, index=False)
    print(f"Saved forecasts to {merged_path}")

if __name__ == "__main__":
    main()
