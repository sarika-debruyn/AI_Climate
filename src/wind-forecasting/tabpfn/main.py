import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# === Step 1: Estimate Wind Power ===
def compute_wind_power(df, rotor_area=1.6, air_density=1.121):
    """
    Estimate wind power (in kW) from wind speed.
    """
    df['wind_power'] = 0.5 * air_density * rotor_area * (df['Wind Speed'] ** 3) / 1000  # convert W to kW
    return df


# === Step 2: Load and Prepare Wind Data ===
def load_wind_data(base_dir="wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    
    dfs = []
    for path in file_paths:
        if not path.exists():
            print(f"Warning: Missing file {path}")
            continue
        print(f"Loading {path}")
        dfs.append(pd.read_csv(path, skiprows=2))

    if not dfs:
        raise ValueError("No wind CSVs were loaded. Check paths or uploads.")

    df = pd.concat(dfs, ignore_index=True)

    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    for col in ['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure', 'Cloud Type', 'Dew Point']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Wind Speed'], inplace=True)

    df = compute_wind_power(df)

    # Feature engineering
    df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    feature_cols = [
        'Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure',
        'Dew Point', 'cos_hour', 'sin_hour', 'dayofyear', 'is_clear'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'wind_power']

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
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Print a preview of predictions
    df_results = pd.DataFrame({
        'y_true': y_test[:10].values,
        'y_pred': y_pred[:10]
    })
    print("\n=== Sample Predictions ===")
    print(df_results)

    return model


# === Step 5: Main Script ===
if __name__ == "__main__":
    X, y = load_wind_data(base_dir="wind_data", years=range(2018, 2024))

    # Sample if above TabPFN size limit
    max_samples = 10_000
    if len(X) > max_samples:
        sampled_indices = np.random.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[sampled_indices]
        y = y.iloc[sampled_indices]

    (X_train, X_test, y_train, y_test), scaler = preprocess(X, y)
    model = train_evaluate_tabpfn(X_train, X_test, y_train, y_test)
