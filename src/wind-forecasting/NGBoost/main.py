import pandas as pd
import numpy as np
from pathlib import Path
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.preprocessing import StandardScaler
import os
import warnings

# === Constants ===
AIR_DENSITY = 1.121  # kg/m^3
TURBINE_RADIUS = 50  # meters (100m diameter)
SWEEP_AREA = np.pi * TURBINE_RADIUS**2
EFFICIENCY = 0.40
TURBINE_COUNT = 16

# === Forecast Horizon ===
FORECAST_START = "2024-01-01"
FORECAST_END = "2024-12-31 23:00"

# === Feature Columns ===
FEATURE_COLUMNS = [
    'sin_hour', 'cos_hour', 'dayofyear',
    'Temperature', 'Pressure', 'Relative Humidity', 'Dew Point', 'is_clear'
]

# === Load Wind Data ===
def load_wind_data(base_dir="../wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    for col in ['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure', 'Cloud Type', 'Dew Point']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Wind Speed'], inplace=True)
    return df.set_index('datetime').sort_index()

# === Convert Wind Speed to Power (MW) ===
def wind_speed_to_power(wind_speed):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA * EFFICIENCY * TURBINE_COUNT
    return coeff * (wind_speed ** 3) / 1_000_000  # Convert W to MW

# === Feature Engineering ===
def prepare_features(df):
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[FEATURE_COLUMNS]
    target = df['Wind Speed']  # Train to predict wind speed (m/s)
    return features, target

# === Create Synthetic Forecast Features for 2024 ===
def generate_2024_features():
    np.random.seed(42)  # <-- Fix randomness for reproducibility

    date_range = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='h')
    df = pd.DataFrame({'datetime': date_range})

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = 1  # Assume mostly clear skies (can randomize later if you want)

    # Temperature: smooth seasonal + random daily noise
    base_temp = 10 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365)
    temp_noise = np.random.normal(0, 2, size=len(df))  # ±2°C daily variability
    df['Temperature'] = base_temp + temp_noise

    # Pressure: around 1013 hPa ± small Gaussian noise
    df['Pressure'] = 1013 + np.random.normal(0, 5, size=len(df))  # ±5 hPa

    # Relative Humidity: higher in winter, lower in summer + noise
    base_rh = 60 - 10 * np.sin(2 * np.pi * df['dayofyear'] / 365)  # Higher RH in winter
    rh_noise = np.random.normal(0, 5, size=len(df))  # ±5% RH
    df['Relative Humidity'] = np.clip(base_rh + rh_noise, 20, 100)

    # Dew Point: rough function of Temperature and RH
    df['Dew Point'] = df['Temperature'] - (100 - df['Relative Humidity']) / 5

    return df.set_index('datetime')


# === Train and Forecast ===
def train_and_forecast(X_train, y_train, X_forecast):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_forecast_scaled = scaler.transform(X_forecast)

    model = NGBRegressor(
        Dist=Normal,
        Score=MLE,
        n_estimators=1294,
        learning_rate=0.026459068629825326,
        minibatch_frac=0.6050081825736404,
        natural_gradient=True,
        verbose=True
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train_scaled, y_train)
        wind_speed_forecast = model.predict(X_forecast_scaled)

    return wind_speed_forecast

# === Main Script ===
def main():
    print("Loading wind data...")
    df = load_wind_data()
    X_train, y_train = prepare_features(df)

    print("Generating synthetic 2024 features...")
    df_forecast = generate_2024_features()
    X_forecast = df_forecast[FEATURE_COLUMNS]

    print("Training and forecasting wind speed for 2024...")
    wind_speed_pred = train_and_forecast(X_train, y_train, X_forecast)

    print("Converting predicted wind speed to power output...")
    wind_power_pred = wind_speed_to_power(wind_speed_pred)

    forecast_df = pd.DataFrame({
        'datetime': df_forecast.index,
        'wind_power_mw': wind_power_pred
    })

    print("Saving forecast results...")
    output_dir = Path("../../model_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_dir / "wind_ngboost_eval_forecast.csv", index=False)
    print("Wind NGBoost 2024 forecast saved to model_results.")

if __name__ == "__main__":
    main()

print(xgboost.__version__)