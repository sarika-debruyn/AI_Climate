import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from pandas.api.types import CategoricalDtype
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Constants ===
AIR_DENSITY = 1.121  # kg/m³
TURBINE_RADIUS = 40  # meters
SWEEP_AREA = np.pi * TURBINE_RADIUS**2

# === Generate 2024 Forecast Timestamps ===
def generate_forecast_timestamps(start="2024-01-01", end="2024-12-31 23:00"):
    return pd.date_range(start=start, end=end, freq="H")

# === Load Historical Wind Data ===
def load_wind_data(base_dir="../wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)
    df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')
    df.dropna(subset=['Wind Speed'], inplace=True)
    return df

# === Estimate Wind Power (W) ===
def estimate_wind_power(df):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA
    df['wind_power_w'] = coeff * (df['Wind Speed'] ** 3)
    return df

# === Climatology Model ===
def train_climatology_model(df):
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    return df.groupby(['month', 'hour'])['wind_power_w'].mean().reset_index().rename(columns={'wind_power_w': 'wind_climatology_w'})

# === Forecast for 2024 using climatology ===
def generate_baseline_forecast(climatology_df):
    forecast_timestamps = generate_forecast_timestamps()
    df = pd.DataFrame({'datetime': forecast_timestamps})
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df = df.merge(climatology_df, on=['month', 'hour'], how='left')
    df['wind_power_mw'] = df['wind_climatology_w'] / 1e6
    return df[['datetime', 'wind_power_mw']].dropna()

# === Main Pipeline ===
def main():
    print("Loading historical wind data...")
    df_hist = load_wind_data()
    df_hist = estimate_wind_power(df_hist)

    print("Training climatology model...")
    climatology_df = train_climatology_model(df_hist)

    print("Generating 2024 baseline forecast...")
    forecast_df = generate_baseline_forecast(climatology_df)

    print("Saving forecast to model_results...")
    os.makedirs("../../model_results", exist_ok=True)
    climatology_df.to_csv("../../model_results/wind_climatology.csv", index=False)
    forecast_df.to_csv("../../model_results/wind_baseline_eval_forecast.csv", index=False)
    print("✅ Saved climatology and baseline forecast for wind.")

if __name__ == "__main__":
    main()
