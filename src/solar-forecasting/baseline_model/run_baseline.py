import pandas as pd
from data_loader import load_solar_data
from solar_power_calc import add_power_output
from climatology_model import train_climatology_model, apply_climatology_model
from evaluate_model import evaluate_forecast
from visualize import plot_forecast, plot_heatmap
from pandas.api.types import CategoricalDtype

# Load and prep data
df = load_solar_data()
df['month'] = df.index.month.astype(CategoricalDtype(categories=list(range(1, 13)), ordered=True))
df['hour'] = df.index.hour.astype(CategoricalDtype(categories=list(range(0, 24)), ordered=True))
df['year'] = df.index.year

# Add solar output calculation
df = add_power_output(df)

# Split train/test
df_train = df[df['year'] <= 2020].copy()
df_test = df[df['year'] >= 2021].copy()

# Train/apply climatology model
climatology = train_climatology_model(df_train)
df_forecast = apply_climatology_model(df_test, climatology)

# Evaluate
mae, rmse = evaluate_forecast(df_forecast)
print(f"\n📊 Climatology MAE (2021–2023): {mae:.2f} W/m²")
print(f"📊 Climatology RMSE (2021–2023): {rmse:.2f} W/m²\n")

# Visualize
plot_forecast(df_forecast)
plot_heatmap(df_forecast, year=2022)
