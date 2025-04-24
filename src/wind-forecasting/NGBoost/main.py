import pandas as pd
import numpy as np
from pathlib import Path
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import os
import warnings

# === Constants ===
AIR_DENSITY = 1.121  # kg/m³
TURBINE_RADIUS = 50  # meters (for 100m diameter)
SWEEP_AREA = np.pi * TURBINE_RADIUS**2  # ~7,850 m²
EFFICIENCY = 0.40
TURBINE_COUNT = 16  # 2.5 MW * 16 = 40 MW total

# === Forecast Horizon ===
FORECAST_START = "2024-01-01"
FORECAST_END = "2024-12-31 23:00"

# === Load Wind Data ===
def load_wind_data(base_dir="../wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    for col in ['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure', 'Cloud Type', 'Dew Point']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Wind Speed'], inplace=True)
    return df.set_index('datetime').sort_index()

# === Convert Wind Speed to Total Wind Farm Power (kW) ===
def wind_speed_to_power(wind_speed):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA * EFFICIENCY * TURBINE_COUNT
    return coeff * (wind_speed ** 3) / 1000  # in kW

# === Feature Engineering ===
def prepare_features(df):
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['sin_hour', 'cos_hour', 'dayofyear',
                   'Temperature', 'Pressure', 'Relative Humidity', 'Dew Point', 'is_clear']]
    target = wind_speed_to_power(df['Wind Speed'])
    return features, target

# === Create Synthetic Forecast Features for 2024 ===
def generate_2024_features():
    date_range = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='h')
    df = pd.DataFrame({'datetime': date_range})
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = 1
    df['Temperature'] = 10 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['Pressure'] = 1013
    df['Relative Humidity'] = 50
    df['Dew Point'] = 5
    return df.set_index('datetime')

# === Train and Forecast ===
def train_and_forecast(X_train, y_train, X_forecast):
    model = NGBRegressor(Dist=Normal, Score=MLE, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train, y_train)
        y_forecast = model.predict(X_forecast)
    return y_forecast

# === Main ===
def main():
    print("Loading wind data...")
    df = load_wind_data()
    X_train, y_train = prepare_features(df)

    print("Generating synthetic 2024 features...")
    df_forecast = generate_2024_features()
    X_forecast = df_forecast

    print("Training and forecasting 2024...")
    y_pred = train_and_forecast(X_train, y_train, X_forecast)
    forecast_df = pd.DataFrame({
        'datetime': df_forecast.index,
        'wind_power_mw': y_pred / 1000  # convert kW to MW for the full farm
    })

    print("Saving forecast results...")
    os.makedirs("../../model_results", exist_ok=True)
    forecast_df.to_csv("../../model_results/wind_ngboost_eval_forecast.csv", index=False)
    print("✅ Wind NGBoost 2024 forecast saved to model_results.")

if __name__ == "__main__":
    main()
