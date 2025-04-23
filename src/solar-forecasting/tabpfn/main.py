import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor


# === Step 1: Estimate Solar Power ===
def compute_solar_power(df, panel_area=1.6, efficiency=0.20):
    """
    Estimate solar power (in kW) from GHI.
    """
    df['solar_power'] = (df['GHI'] * panel_area * efficiency) / 1000  # convert W to kW
    return df


# === Step 2: Load and Prepare Solar Data ===
def load_solar_data(base_dir="/Users/sarikadebruyn/AI_Climate/src/solar-forecasting/solar_data/", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    
    # Load all CSVs (skipping first 2 rows if needed)
    dfs = []
    for path in file_paths:
        if not path.exists():
            print(f"Warning: Missing file {path}")
            continue
        dfs.append(pd.read_csv(path, skiprows=2))

    df = pd.concat(dfs, ignore_index=True)

    # Create datetime
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    # Clean numeric values
    for col in ['GHI', 'DHI', 'DNI', 'Temperature', 'Wind Speed', 'Relative Humidity', 'Cloud Type']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop missing GHI rows (required for power calc)
    df.dropna(subset=['GHI'], inplace=True)

    # Compute solar power from GHI
    df = compute_solar_power(df)

    # Feature engineering
    df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['dayofyear'] = df['datetime'].dt.dayofyear

    # Feature selection
    feature_cols = [
        'GHI', 'Temperature', 'DHI', 'DNI',
        'Wind Speed', 'Relative Humidity', 'Pressure',
        'Ozone', 'Dew Point', 'cos_hour', 'sin_hour', 'dayofyear'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df['solar_power']

    # Drop any remaining NaNs
    X = X.dropna()
    y = y.loc[X.index]

    return X, y


# === Step 3: Preprocess (Standardize & Split) ===
def preprocess(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler


# === Step 4: Train and Evaluate TabPFN ===
def train_evaluate_tabpfn(X_train, X_test, y_train, y_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model = TabPFNRegressor(device=device)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return model


# === Step 5: Main Script ===
if __name__ == "__main__":
    X, y = load_solar_data()
    # Sample if above TabPFN size limit
    #max_samples = 10_000
    #if len(X) > max_samples:
        #sampled_indices = np.random.choice(len(X), size=max_samples, replace=False)
        #X = X.iloc[sampled_indices]
        #y = y.iloc[sampled_indices]
    (X_train, X_test, y_train, y_test), scaler = preprocess(X, y)
    model = train_evaluate_tabpfn(X_train, X_test, y_train, y_test)
