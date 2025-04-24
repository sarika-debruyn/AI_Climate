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
import pvlib

# === Constants ===
LATITUDE = 32.7
LONGITUDE = -114.63
TIMEZONE = 'Etc/GMT+7'
PANEL_AREA = 1.6  # m²
EFFICIENCY_BASE = 0.20
TEMP_COEFF = 0.004
T_REF = 25

# === Forecast Horizon ===
FORECAST_START = "2024-01-01"
FORECAST_END = "2024-12-31 23:00"

# === Load Solar Data ===
def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    for col in ['GHI', 'DHI', 'DNI', 'Temperature', 'Wind Speed', 'Relative Humidity', 'Cloud Type']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['GHI'], inplace=True)
    return df.set_index('datetime').sort_index()

# === Add Solar Zenith ===
def add_zenith_angle(df):
    solar_position = pvlib.solarposition.get_solarposition(
        time=df.index, latitude=LATITUDE, longitude=LONGITUDE
    )
    df['zenith'] = solar_position['zenith'].values
    return df

# === Calculate Solar Power ===
def temp_derated_efficiency(temp, base_eff=EFFICIENCY_BASE, gamma=TEMP_COEFF, T_ref=T_REF):
    return base_eff * (1 - gamma * (temp - T_ref))

def ghi_to_power(ghi, temp):
    eff = temp_derated_efficiency(temp)
    return ghi * PANEL_AREA * eff / 1000

# === Feature Engineering ===
def prepare_features(df):
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['zenith_norm'] = df['zenith'] / 90.0
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['sin_hour', 'cos_hour', 'dayofyear', 'zenith_norm',
                   'DHI', 'DNI', 'Temperature', 'Relative Humidity', 'Wind Speed', 'is_clear']]
    target = ghi_to_power(df['GHI'], df['Temperature'])
    return features, target

# === Create Synthetic Forecast Features for 2024 ===
def generate_2024_features():
    date_range = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='h')
    df = pd.DataFrame({'datetime': date_range})
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['zenith'] = 45  # static placeholder
    df['zenith_norm'] = df['zenith'] / 90.0
    df['is_clear'] = 1
    df['DHI'] = 100
    df['DNI'] = 600
    df['Temperature'] = 25 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['Relative Humidity'] = 40
    df['Wind Speed'] = 3
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
    print("Loading solar data...")
    df = load_solar_data()
    df = add_zenith_angle(df)
    X_train, y_train = prepare_features(df)

    print("Generating synthetic 2024 features...")
    df_forecast = generate_2024_features()
    X_forecast = df_forecast

    print("Training and forecasting 2024...")
    y_pred = train_and_forecast(X_train, y_train, X_forecast)
    forecast_df = pd.DataFrame({
        'datetime': df_forecast.index,
        'solar_power_mw': y_pred / 1000
    })

    print("Saving forecast results...")
    os.makedirs("../../model_results", exist_ok=True)
    forecast_df.to_csv("../../model_results/solar_ngboost_eval_forecast.csv", index=False)
    print("✅ Solar NGBoost 2024 forecast saved to model_results.")

if __name__ == "__main__":
    main()
