import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.prep_data import load_and_clean_data
from scripts.climatology import climatology_forecast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


base_dir = "src/solar-forecasting/solar_data"
file_names = [
    "solar_2018.csv", "solar_2019.csv", "solar_2020.csv",
    "solar_2021.csv", "solar_2022.csv", "solar_2023.csv"
]

file_paths = [os.path.join(base_dir, name) for name in file_names]

df_ready = load_and_clean_data(file_paths)
df_forecast = climatology_forecast(df_ready)

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI'][-500:], label='Actual')
plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI_climatology'][-500:], label='Climatology', linestyle='--')
plt.legend()
plt.title("Climatology Forecast vs Actual")
plt.show()

# Metrics
mae = mean_absolute_error(df_forecast['GHI'], df_forecast['GHI_climatology'])
rmse = np.sqrt(mean_squared_error(df_forecast['GHI'], df_forecast['GHI_climatology']))
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
