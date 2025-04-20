import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pandas.api.types import CategoricalDtype
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Config ===
AIR_DENSITY = 1.121  # kg/m^3 for Mount Storm, WV
turbine_radius = 40  # meters
SWEEP_AREA = np.pi * turbine_radius**2

# === Load Data ===
def load_wind_data(base_dir="../wind_data"):
    all_years = [2018, 2019, 2020, 2021, 2022, 2023]
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in all_years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)
    df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')
    df.dropna(subset=['Wind Speed'], inplace=True)
    return df

# === Estimate Wind Power ===
def estimate_wind_power(df):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA
    df['Wind Power (W)'] = coeff * (df['Wind Speed'] ** 3)
    return df

# === Create Climatology Model ===
def create_climatology_model(df):
    df['year'] = df.index.year
    df['month'] = df.index.month.astype(CategoricalDtype(categories=list(range(1, 13)), ordered=True))
    df['hour'] = df.index.hour.astype(CategoricalDtype(categories=list(range(0, 24)), ordered=True))
    df_train = df[df['year'] <= 2020].copy()
    df_test = df[df['year'] >= 2021].copy()
    climatology = df_train.groupby(['month', 'hour'])['Wind Power (W)'].mean().reset_index()
    climatology.rename(columns={'Wind Power (W)': 'WindPower_climatology'}, inplace=True)
    df_test_reset = df_test.reset_index()
    df_forecast = pd.merge(df_test_reset, climatology, on=['month', 'hour'], how='left')
    return df_forecast, df_test

# === Evaluate Model ===
def evaluate_model(df_forecast):
    mae = mean_absolute_error(df_forecast['Wind Power (W)'], df_forecast['WindPower_climatology'])
    rmse = np.sqrt(mean_squared_error(df_forecast['Wind Power (W)'], df_forecast['WindPower_climatology']))
    print(f"Climatology MAE (2021–2023): {mae:.2f} W")
    print(f"Climatology RMSE (2021–2023): {rmse:.2f} W")
    return mae, rmse

# === Visualization ===
def plot_forecast(df_forecast):
    plt.figure(figsize=(15, 4))
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['Wind Power (W)'][-500:], label='Actual Wind Power')
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['WindPower_climatology'][-500:], label='Climatology Forecast', linestyle='--')
    plt.title('Climatology Forecast vs Actual (Last 500 Hours of Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Estimated Wind Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_forecast):
    df_heat = df_forecast[df_forecast['timestamp'].dt.year == 2022].copy()
    df_heat['abs_error'] = abs(df_heat['Wind Power (W)'] - df_heat['WindPower_climatology'])
    heatmap_data = df_heat.groupby(['month', 'hour'])['abs_error'].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5, linecolor='gray')
    plt.title("Climatology Forecast MAE by Month & Hour (2022 Only)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.show()

# === Run Baseline ===
if __name__ == "__main__":
    df = load_wind_data()
    df = estimate_wind_power(df)
    df_forecast, df_test = create_climatology_model(df)
    evaluate_model(df_forecast)
    plot_forecast(df_forecast)
    plot_heatmap(df_forecast)
