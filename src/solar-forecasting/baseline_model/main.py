
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# === Data Loading ===
def load_solar_data(base_dir="/Users/sarikadebruyn/AI_Climate/AI_Climate/src/solar-forecasting/solar_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df.dropna(subset=['GHI', 'Temperature'], inplace=True)
    return df

# === Solar Zenith and Power Output Calculation ===
def add_zenith_angle(df, latitude=LATITUDE, longitude=LONGITUDE):
    df = df.copy()
    solar_position = pvlib.solarposition.get_solarposition(
        time=df.index, latitude=latitude, longitude=longitude
    )
    df['zenith'] = solar_position['zenith'].values
    return df

def temp_derated_efficiency(temp, base_eff=EFFICIENCY_BASE, gamma=TEMP_COEFF, T_ref=T_REF):
    return base_eff * (1 - gamma * (temp - T_ref))

def add_power_output(df):
    df = add_zenith_angle(df)
    eff = temp_derated_efficiency(df['Temperature'])
    df['solar_output_W'] = df['GHI'] * PANEL_AREA * eff
    df['solar_output_kW'] = df['solar_output_W'] / 1000
    return df

# === Climatology Model ===
def train_climatology_model(df_train):
    climatology = df_train.groupby(['month', 'hour'])['GHI'].mean().reset_index()
    climatology.rename(columns={'GHI': 'GHI_climatology'}, inplace=True)
    return climatology

def apply_climatology_model(df_test, climatology):
    df_test = df_test.reset_index()
    df_forecast = pd.merge(df_test, climatology, on=['month', 'hour'], how='left')
    return df_forecast

# === Evaluation ===
def evaluate_forecast(df_forecast):
    mae = mean_absolute_error(df_forecast['GHI'], df_forecast['GHI_climatology'])
    rmse = np.sqrt(mean_squared_error(df_forecast['GHI'], df_forecast['GHI_climatology']))
    return mae, rmse

# === Visualization ===
def plot_forecast(df_forecast):
    plt.figure(figsize=(15, 4))
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI'][-500:], label='Actual GHI')
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI_climatology'][-500:], label='Climatology Forecast', linestyle='--')
    plt.title('Climatology Forecast vs Actual (Last 500 Hours)')
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_forecast, year=2022):
    df = df_forecast[df_forecast['timestamp'].dt.year == year].copy()
    df['abs_error'] = abs(df['GHI'] - df['GHI_climatology'])
    heatmap_data = df.groupby(['month', 'hour'])['abs_error'].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5, linecolor='gray')
    plt.title(f"Climatology Forecast MAE by Month & Hour ({year} Only)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.show()

# === Main Pipeline ===
def main():
    df = load_solar_data()
    df['month'] = df.index.month.astype(CategoricalDtype(categories=list(range(1, 13)), ordered=True))
    df['hour'] = df.index.hour.astype(CategoricalDtype(categories=list(range(0, 24)), ordered=True))
    df['year'] = df.index.year

    df = add_power_output(df)

    df_train = df[df['year'] <= 2020].copy()
    df_test = df[df['year'] >= 2021].copy()

    climatology = train_climatology_model(df_train)
    df_forecast = apply_climatology_model(df_test, climatology)

    mae, rmse = evaluate_forecast(df_forecast)
    print(f"\n📊 Climatology MAE (2021–2023): {mae:.2f} W/m²")
    print(f"📊 Climatology RMSE (2021–2023): {rmse:.2f} W/m²\n")

    plot_forecast(df_forecast)
    plot_heatmap(df_forecast, year=2022)

if __name__ == "__main__":
    main()
