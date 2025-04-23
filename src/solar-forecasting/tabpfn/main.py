# === TabPFN Solar Forecast with Timestamp Output (Fixed Index Issue + Results Directory Creation) ===
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# === Step 1: Estimate Solar Power ===
def compute_solar_power(df, panel_area=1.6, efficiency=0.20):
    df['solar_power'] = (df['GHI'] * panel_area * efficiency) / 1000  # kW
    return df

# === Step 2: Load and Prepare Solar Data ===
def load_solar_data(base_dir="solar_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    dfs = []
    for path in file_paths:
        if not path.exists():
            print(f"Warning: Missing file {path}")
            continue
        print(f"Loading {path}")
        dfs.append(pd.read_csv(path, skiprows=2))

    if not dfs:
        raise ValueError("No solar CSVs were loaded. Check paths or uploads.")

    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    for col in ['GHI', 'DHI', 'DNI', 'Temperature', 'Wind Speed', 'Relative Humidity', 'Pressure', 'Ozone', 'Dew Point']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['GHI'], inplace=True)

    df = compute_solar_power(df)
    df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['dayofyear'] = df['datetime'].dt.dayofyear

    feature_cols = [
        'GHI', 'Temperature', 'DHI', 'DNI',
        'Wind Speed', 'Relative Humidity', 'Pressure',
        'Ozone', 'Dew Point', 'cos_hour', 'sin_hour', 'dayofyear'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    y = df['solar_power']
    timestamps = df['datetime']
    return X, y, timestamps

# === Step 3: Preprocess ===
def preprocess(X, y, timestamps):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X_scaled, y, timestamps, test_size=0.2, random_state=42
    )
    return (X_train, X_test, y_train, y_test, t_train, t_test), scaler

# === Step 4: Train and Evaluate ===
def train_evaluate_tabpfn(X_train, X_test, y_train, y_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model = TabPFNRegressor(device=device)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return y_pred

# === Step 5: Main Script ===
if __name__ == "__main__":
    X, y, timestamps = load_solar_data()

    max_samples = 10_000
    if len(X) > max_samples:
        sampled_indices = np.random.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[sampled_indices]
        y = y.iloc[sampled_indices]
        timestamps = timestamps.iloc[sampled_indices]

    (X_train, X_test, y_train, y_test, t_train, t_test), scaler = preprocess(X, y, timestamps)
    y_pred = train_evaluate_tabpfn(X_train, X_test, y_train, y_test)

    forecast_df = pd.DataFrame({
        'datetime': t_test.values,
        'solar_power_mw': y_pred / 1000
    }).sort_values(by='datetime')

    os.makedirs("../../results", exist_ok=True)
    forecast_df.to_csv("../../results/solar_tabpfn_forecast.csv", index=False)
    print("Saved TabPFN forecast to ../../results/solar_tabpfn_forecast.csv")
