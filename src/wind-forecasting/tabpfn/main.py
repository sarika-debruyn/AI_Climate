import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import os
import warnings

# === Constants ===
AIR_DENSITY = 1.121  # kg/m³
TURBINE_RADIUS = 50  # meters (100m diameter)
SWEEP_AREA = np.pi * TURBINE_RADIUS**2  # ~7,850 m²
EFFICIENCY = 0.40
TURBINE_COUNT = 16  # 2.5 MW each x 16 turbines = 40 MW
FORECAST_START = "2024-01-01"
FORECAST_END = "2024-12-31 23:00"
MAX_TABPFN_SAMPLES = 10000

# === Load Wind Data ===
def load_wind_data(base_dir="wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    dfs = []
    for path in file_paths:
        if not path.exists():
            print(f"Missing: {path}")
            continue
        print(f"Loaded: {path}")
        dfs.append(pd.read_csv(path, skiprows=2))

    if not dfs:
        raise ValueError("No wind CSVs were loaded. Check paths or uploads.")
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    for col in ['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure', 'Cloud Type', 'Dew Point']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Wind Speed'], inplace=True)
    return df.set_index('datetime').sort_index()

# === Wind Power Estimation (kW) ===
def wind_speed_to_power(wind_speed):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA * EFFICIENCY * TURBINE_COUNT
    return coeff * (wind_speed ** 3) / 1000

# === Feature Engineering ===
def prepare_features(df):
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure',
                   'Dew Point', 'cos_hour', 'sin_hour', 'dayofyear', 'is_clear']]
    target = wind_speed_to_power(df['Wind Speed'])
    return features, target

# === Synthetic 2024 Forecast Features ===
def generate_2024_features():
    date_range = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='h')
    df = pd.DataFrame({'datetime': date_range})
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = 1
    df['Wind Speed'] = 3
    df['Temperature'] = 15 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['Relative Humidity'] = 60
    df['Pressure'] = 1013
    df['Dew Point'] = 10

    return df.set_index('datetime')[['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure',
                                     'Dew Point', 'cos_hour', 'sin_hour', 'dayofyear', 'is_clear']]

# === Train & Forecast ===
def train_and_forecast(X_train, y_train, X_forecast):
    if len(X_train) > MAX_TABPFN_SAMPLES:
        sampled_idx = np.random.choice(len(X_train), size=MAX_TABPFN_SAMPLES, replace=False)
        X_train = X_train.iloc[sampled_idx]
        y_train = y_train.iloc[sampled_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_forecast_scaled = scaler.transform(X_forecast)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabPFNRegressor(device=device)
    model.fit(X_train_scaled, y_train)
    y_forecast = model.predict(X_forecast_scaled)
    return y_forecast

# === Main ===
def main():
    print("Loading wind data...")
    df = load_wind_data()
    X_train, y_train = prepare_features(df)

    print("Generating 2024 forecast features...")
    X_forecast = generate_2024_features()

    print("Training TabPFN and forecasting...")
    y_pred = train_and_forecast(X_train, y_train, X_forecast)

    forecast_df = pd.DataFrame({
        'datetime': X_forecast.index,
        'wind_power_mw': y_pred / 1000  # convert kW to MW
    })

    print("Saving forecast results...")
    os.makedirs("model_results", exist_ok=True)
    forecast_df.to_csv("model_results/wind_tabpfn_eval_forecast.csv", index=False)

    print("Wind TabPFN 2024 forecast saved to model_results.")

if __name__ == "__main__":
    main()
