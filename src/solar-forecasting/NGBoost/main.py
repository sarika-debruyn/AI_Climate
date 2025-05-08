import pandas as pd
import numpy as np
from pathlib import Path
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
import os
import warnings
import pvlib

# === Constants ===
LATITUDE = 32.7
LONGITUDE = -114.63
TIMEZONE = 'Etc/GMT+7'
PANEL_AREA_TOTAL = 256_000  # m²
EFFICIENCY_BASE = 0.20
PERFORMANCE_RATIO = 0.8

# === Feature Columns ===
FEATURE_COLUMNS = [
    'sin_hour', 'cos_hour', 'dayofyear', 'zenith_norm',
    'Temperature', 'Relative Humidity', 'Dew Point', 'is_clear'
]

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
def ghi_to_power(ghi_w_per_m2, zenith_deg):
    performance_correction = np.cos(np.radians(zenith_deg))
    adjusted_ghi = ghi_w_per_m2 * performance_correction
    return (adjusted_ghi * PANEL_AREA_TOTAL * EFFICIENCY_BASE * PERFORMANCE_RATIO) / 1_000_000  # MW

# === Feature Engineering ===
def prepare_features(df):
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['zenith_norm'] = df['zenith'] / 90.0
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[FEATURE_COLUMNS]
    target = df['GHI']  # Train NGBoost to predict GHI (W/m²)
    return features, target

# === Create Synthetic Forecast Features for 2024 ===
def generate_2024_features():
    np.random.seed(42)  # Fix randomness
    date_range = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='h')
    df = pd.DataFrame({'datetime': date_range})
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['zenith'] = 45 + 15 * np.sin(2 * np.pi * df['dayofyear'] / 365)  # Dynamic zenith
    df['zenith_norm'] = df['zenith'] / 90.0
    df['is_clear'] = 1
    df['Temperature'] = 25 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365) + np.random.normal(0, 2, len(df))
    df['Relative Humidity'] = np.clip(40 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365) + np.random.normal(0, 5, len(df)), 20, 100)
    df['Dew Point'] = df['Temperature'] - (100 - df['Relative Humidity']) / 5
    return df.set_index('datetime')

# === Train and Forecast ===
def train_and_forecast(X_train, y_train, X_forecast):

    model = NGBRegressor(Dist=Normal, Score=MLE, verbose=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train, y_train)
        ghi_forecast = model.predict(X_forecast)

    return ghi_forecast

# === Main ===
def main():
    print("Loading solar data...")
    df = load_solar_data()
    df = add_zenith_angle(df)
    X_train, y_train = prepare_features(df)

    print("Generating synthetic 2024 features...")
    df_forecast = generate_2024_features()
    X_forecast = df_forecast[FEATURE_COLUMNS]

    print("Training and forecasting GHI for 2024...")
    ghi_pred = train_and_forecast(X_train, y_train, X_forecast)

    print("Converting predicted GHI to power output...")
    solar_power_pred = ghi_to_power(ghi_pred, df_forecast['zenith'])

    forecast_df = pd.DataFrame({
        'datetime': df_forecast.index,
        'solar_power_mw': solar_power_pred
    })

    print("Saving forecast results...")
    output_dir = Path("../../model_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_dir / "solar_ngboost_eval_forecast.csv", index=False)
    print("Solar NGBoost 2024 forecast saved to model_results.")

if __name__ == "__main__":
    main()
