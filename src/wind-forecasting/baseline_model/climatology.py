# File: src/wind-forecasting/baseline_model/climatology.py
#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from shared.path_utils import WIND_DATA_DIR, wind_output, _ensure_dirs

warnings.filterwarnings("ignore")
_ensure_dirs()  # make sure wind output & visuals folders exist

# === Constants ===
AIR_DENSITY    = 1.121    # kg/m³
TURBINE_RADIUS = 50       # m
SWEEP_AREA     = np.pi * TURBINE_RADIUS**2  # m²
TURBINE_COUNT  = 16       # turbines per farm
efficiency     = 0.40     # turbine efficiency

TRAIN_YEARS   = range(2018, 2023)  # Use 2018–2022 for climatology
HOLDOUT_YEAR  = 2023

# === Helper Functions ===

def load_wind_data(base_dir=WIND_DATA_DIR, years=range(2018, 2024)):
    parts = []
    for yr in years:
        path = Path(base_dir) / f"wind_{yr}.csv"
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        df = pd.read_csv(path, skiprows=2)
        df['datetime'] = pd.to_datetime(
            dict(year=df.Year, month=df.Month, day=df.Day,
                 hour=df.Hour, minute=df.Minute)
        )
        df.set_index('datetime', inplace=True)
        df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')
        parts.append(df.dropna(subset=['Wind Speed']))
    return pd.concat(parts).sort_index()


def wind_speed_to_power(windspeed):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA * efficiency * TURBINE_COUNT
    # returns power in MW
    return coeff * (windspeed ** 3) / 1e6

# === Main ===

def main():

    # 1. Load 2018–2023 data
    df = load_wind_data()

    # 2. Compute farm-level power
    df['wind_power_MW'] = wind_speed_to_power(df['Wind Speed'])

    # 3. Split training (2018–2022) and hold-out (2023)
    df_train = df[df.index.year.isin(TRAIN_YEARS)]
    df_hold  = df[df.index.year == HOLDOUT_YEAR]

    # 4. Compute climatology: mean wind_power by (month, hour)
    df_train['month'] = df_train.index.month
    df_train['hour']  = df_train.index.hour
    climatology = (
        df_train
        .groupby(['month','hour'])['wind_power_MW']
        .mean()
        .reset_index()
        .rename(columns={'wind_power_MW':'clim_power_MW'})
    )
    climatology.to_csv(
        wind_output("wind_climatology_profile.csv"), index=False
    )

    # 5. Forecast for 2023 based on climatology
    df_hold['month'] = df_hold.index.month
    df_hold['hour']  = df_hold.index.hour
    df_fore = (
        df_hold[['month','hour']]
        .merge(climatology, on=['month','hour'], how='left')
    )
    df_fore['true_power_MW'] = df_hold['wind_power_MW'].values
    df_fore['forecast_power_MW'] = df_fore['clim_power_MW']

    # 6. Evaluate RMSE
    rmse_power = mean_squared_error(
        df_fore['true_power_MW'], df_fore['forecast_power_MW']
    )
    rmse_df = pd.DataFrame([{'metric':'RMSE_power_MW', 'value': rmse_power}])
    rmse_df.to_csv(
        wind_output("wind_climatology_2023_rmse.csv"), index=False
    )

    # 7. Save forecasts
    df_out = df_fore[['true_power_MW','forecast_power_MW']].copy()
    df_out.index = df_hold.index
    df_out.index.name = 'datetime'
    df_out.to_csv(
        wind_output("wind_climatology_2023_forecast.csv")
    )

    print("Wind climatology baseline completed:")
    print(" - Profile: model_results/wind_climatology_profile.csv")
    print(" - Forecast: model_results/wind_climatology_2023_forecast.csv")
    print(" - RMSE: model_results/wind_climatology_2023_rmse.csv")

if __name__ == '__main__':
    main()
