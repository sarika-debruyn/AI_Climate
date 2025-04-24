import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.api.types import CategoricalDtype
import pvlib

# === Constants ===
LATITUDE = 32.7
LONGITUDE = -114.63
TIMEZONE = 'Etc/GMT+7'
PANEL_AREA = 1.6  # m²
EFFICIENCY_BASE = 0.20
TEMP_COEFF = 0.004
T_REF = 25
TOTAL_FARM_AREA = 256000  # m² for 40 MW farm

# === Generate 2024 Forecast Timestamps ===
def generate_forecast_timestamps(start="2024-01-01", end="2024-12-31 23:00"):
    return pd.date_range(start=start, end=end, freq="H")

# === Load Historical Data ===
def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df.dropna(subset=['GHI', 'Temperature'], inplace=True)
    return df

# === Climatology Model ===
def train_climatology_model(df):
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    return df.groupby(['month', 'hour'])['GHI'].mean().reset_index().rename(columns={'GHI': 'GHI_climatology'})

# === Forecast for 2024 using climatology ===
def generate_baseline_forecast(climatology_df):
    forecast_timestamps = generate_forecast_timestamps()
    df = pd.DataFrame({'datetime': forecast_timestamps})
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df = df.merge(climatology_df, on=['month', 'hour'], how='left')
    df['solar_power_mw'] = (df['GHI_climatology'] * TOTAL_FARM_AREA * EFFICIENCY_BASE) / 1000
    return df[['datetime', 'solar_power_mw']].dropna()

# === Main Pipeline ===
def main():
    print("Loading historical solar data...")
    df_hist = load_solar_data()

    print("Training climatology model...")
    climatology_df = train_climatology_model(df_hist)

    print("Generating 2024 baseline forecast...")
    forecast_df = generate_baseline_forecast(climatology_df)

    print("Saving forecast to model_results...")
    os.makedirs("../../model_results", exist_ok=True)
    climatology_df.to_csv("../../model_results/solar_climatology.csv", index=False)
    forecast_df.to_csv("../../model_results/solar_baseline_eval_forecast.csv", index=False)
    print("✅ Saved climatology and baseline forecast for solar.")

if __name__ == "__main__":
    main()
