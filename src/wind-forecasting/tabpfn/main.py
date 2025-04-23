import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# === Constants ===
AIR_DENSITY = 1.121  # kg/m³
ROTOR_AREA = 1.6     # m²
MAX_SAMPLES = 10_000

# === Step 1: Load wind data ===
def load_wind_data(base_dir="wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    dfs = []

    for path in file_paths:
        if not path.exists():
            print(f"Missing file: {path}")
            continue
        dfs.append(pd.read_csv(path, skiprows=2))

    if not dfs:
        raise ValueError("No wind CSV files found.")

    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    for col in ['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure', 'Cloud Type', 'Dew Point']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Wind Speed'], inplace=True)
    return df

# === Step 2: Convert Wind Speed to Power (kW) ===
def wind_speed_to_power(wind_speed, rho=AIR_DENSITY, area=ROTOR_AREA):
    return 0.5 * rho * area * (wind_speed ** 3) / 1000

# === Step 3: Feature engineering ===
def prepare_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['sin_hour', 'cos_hour', 'dayofyear',
                   'Temperature', 'Pressure', 'Relative Humidity', 'Dew Point', 'is_clear']]

    target = wind_speed_to_power(df['Wind Speed'])
    features = features.dropna()
    target = target.loc[features.index]

    return features, target

# === Step 4: Preprocess (scale + split) ===
def preprocess(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# === Step 5: Train and Evaluate TabPFN ===
def train_evaluate_tabpfn(X_train, X_test, y_train, y_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model = TabPFNRegressor(device=device)
    model.fit(X_train, y_train, ignore_pretraining_limits=True)  # <-- fixed here
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\n=== Sample Predictions ===")
    print(pd.DataFrame({'y_true': y_test[:10].values, 'y_pred': y_pred[:10]}))

    return model

# === Main Execution ===
if __name__ == "__main__":
    df = load_wind_data(base_dir="wind_data", years=range(2018, 2024))
    X, y = prepare_features(df)

    if len(X) > MAX_SAMPLES:
        sampled_idx = np.random.choice(X.index, size=MAX_SAMPLES, replace=False)
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]

    (X_train, X_test, y_train, y_test), scaler = preprocess(X, y)
    model = train_evaluate_tabpfn(X_train, X_test, y_train, y_test)
