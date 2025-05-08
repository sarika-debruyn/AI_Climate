import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

# === Constants ===
PANEL_AREA_TOTAL   = 256_000    # m² (40 MW solar farm)
EFFICIENCY_BASE    = 0.20       # Nominal panel efficiency
PERFORMANCE_RATIO  = 0.8        # System performance ratio

TRAIN_YEARS   = range(2018, 2023)  # Use 2018–2022 for climatology
HOLDOUT_YEAR  = 2023

# === Helper Functions ===

def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    parts = []
    for yr in years:
        path = Path(base_dir) / f"solar_{yr}.csv"
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        df = pd.read_csv(path, skiprows=2)
        df['datetime'] = pd.to_datetime(
            dict(year=df.Year, month=df.Month, day=df.Day,
                 hour=df.Hour, minute=df.Minute)
        )
        df.set_index('datetime', inplace=True)
        df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
        parts.append(df.dropna(subset=['GHI']))
    return pd.concat(parts).sort_index()


def ghi_to_power(ghi):
    """
    Convert GHI (W/m²) to farm-level power in MW.
    """
    return (ghi * PANEL_AREA_TOTAL * EFFICIENCY_BASE * PERFORMANCE_RATIO) / 1e6

# === Main ===

def main():
    os.makedirs("model_results", exist_ok=True)

    # 1. Load data for 2018–2023
    df = load_solar_data()

    # 2. Split into training (2018–2022) and hold-out (2023)
    df_train = df[df.index.year.isin(TRAIN_YEARS)]
    df_hold  = df[df.index.year == HOLDOUT_YEAR]

    # 3. Compute climatology: mean GHI by (month, hour)
    df_train['month'] = df_train.index.month
    df_train['hour']  = df_train.index.hour
    climatology = (
        df_train
        .groupby(['month','hour'])['GHI']
        .mean()
        .reset_index()
        .rename(columns={'GHI':'GHI_climatology'})
    )
    climatology.to_csv(
        "model_results/solar_climatology_profile.csv", index=False
    )

    # 4. Forecast for 2023 hold-out using climatology
    df_hold = df_hold.copy()
    df_hold['month'] = df_hold.index.month
    df_hold['hour']  = df_hold.index.hour
    df_fore = (
        df_hold
        .merge(climatology, on=['month','hour'], how='left')
    )
    # Convert to power
    df_fore['power_true_MW']  = ghi_to_power(df_fore['GHI'])
    df_fore['power_clim_MW']  = ghi_to_power(df_fore['GHI_climatology'])

    # 5. Evaluate RMSE
    rmse_ghi   = mean_squared_error(
        df_fore['GHI'], df_fore['GHI_climatology']
    )
    rmse_power = mean_squared_error(
        df_fore['power_true_MW'], df_fore['power_clim_MW']
    )
    rmse_df = pd.DataFrame([
        {'metric':'RMSE_GHI', 'value':rmse_ghi},
        {'metric':'RMSE_power_MW', 'value':rmse_power}
    ])
    rmse_df.to_csv(
        "model_results/solar_climatology_2023_rmse.csv", index=False
    )

    # 6. Save hold-out forecasts
    df_out = df_fore[['GHI','GHI_climatology','power_true_MW','power_clim_MW']].copy()
    df_out.index.name = 'datetime'
    df_out.to_csv(
        "model_results/solar_climatology_2023_forecast.csv"
    )

    print("Climatology baseline completed:")
    print(" - Profile: model_results/solar_climatology_profile.csv")
    print(" - Forecast: model_results/solar_climatology_2023_forecast.csv")
    print(" - RMSE: model_results/solar_climatology_2023_rmse.csv")

if __name__ == '__main__':
    main()
