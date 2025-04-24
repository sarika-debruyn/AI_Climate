import xarray as xr
import pandas as pd
import numpy as np

# === CONFIGURATION ===
filename = "solar_data/solar_2024_part1.nc"  
panel_area = 1.6  # in m² (typical single solar panel)
efficiency = 0.20  # panel efficiency
timezone_offset = -7  # For Yuma, AZ (UTC−7)

# === LOAD DATA ===
ds = xr.open_dataset(filename)

# === PRINT VARIABLE NAMES IF NEEDED ===
print("Available variables:", ds.data_vars.keys())

# === EXTRACT VARIABLES ===
ghi = ds['ssrd']  # Surface solar radiation downward (J/m^2)
temp = ds['t2m']  # 2m temperature (Kelvin)
times = pd.to_datetime(ds['time'].values)

# Convert GHI from J/m² to W/m² by dividing by 3600 (J/s = Watts) — hourly average
ghi_w_m2 = ghi.values[:, 0, 0] / 3600

# Convert temperature from Kelvin to Celsius
temp_c = temp.values[:, 0, 0] - 273.15

# === CALCULATE SOLAR POWER OUTPUT (kW) ===
solar_power_kw = (ghi_w_m2 * panel_area * efficiency) / 1000  # in kW

# === CREATE DATAFRAME ===
df = pd.DataFrame({
    'timestamp_utc': times,
    'ghi_w_m2': ghi_w_m2,
    'temperature_c': temp_c,
    'solar_power_kw': solar_power_kw
})

# === OPTIONAL: ADJUST TIMEZONE (if comparing to local datasets) ===
df['timestamp_local'] = df['timestamp_utc'] + pd.to_timedelta(timezone_offset, unit='h')

# === SAVE OR RETURN ===
df.to_csv("solar_power_ground_truth_2024.csv", index=False)
print("Saved to: solar_power_ground_truth_2024.csv")

# === Preview ===
print(df.head())
