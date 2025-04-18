import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# === Load and clean data ===
base_dir = Path("../solar_data")  # adjust as needed
file_paths = [base_dir / f"solar_{year}.csv" for year in range(2018, 2024)]

df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)

df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Day'] = df['Day'].astype(int)
df['Hour'] = df['Hour'].astype(int)
df['Minute'] = df['Minute'].astype(int)

df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df.set_index('timestamp', inplace=True)

df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
df = df.dropna(subset=['GHI'])

df['month'] = df.index.month
df['hour'] = df.index.hour

# === Climatology forecast ===
climatology = df.groupby(['month', 'hour'])['GHI'].mean().reset_index()
climatology.rename(columns={'GHI': 'GHI_climatology'}, inplace=True)

df_reset = df.reset_index()
df_forecast = pd.merge(df_reset, climatology, on=['month', 'hour'], how='left')

# === Plot ===
plt.figure(figsize=(15, 5))
plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI'][-500:], label='Actual GHI')
plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI_climatology'][-500:], label='Climatology Forecast', linestyle='--')
plt.title('Climatology Forecast vs Actual (Last 500 Hours)')
plt.xlabel('Time')
plt.ylabel('GHI')
plt.legend()
plt.tight_layout()
plt.show()

# === Evaluate ===
mae = mean_absolute_error(df_forecast['GHI'], df_forecast['GHI_climatology'])
rmse = np.sqrt(mean_squared_error(df_forecast['GHI'], df_forecast['GHI_climatology']))
print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f}")
