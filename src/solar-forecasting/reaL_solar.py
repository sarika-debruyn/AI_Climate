import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
data_folder = Path("solar_data/")  
panel_area = 1.6  # m²
efficiency = 0.20
timezone_offset = -7  # Yuma, AZ timezone offset (UTC−7)

# === FIND ALL NETCDF FILES IN FOLDER ===
nc_files = sorted(data_folder.glob("*.nc"))
print("Looking in:", data_folder)
print("Found files:", nc_files)

if not nc_files:
    raise FileNotFoundError("⚠️ No .nc files found in the specified folder. Check your path.")

# === LOAD AND CONCATENATE ===
datasets = [xr.open_dataset(file) for file in nc_files]
ds_merged = xr.concat(datasets, dim="time")

# After merging the datasets
print("Available variables:")
print(ds_merged.data_vars.keys())


# === EXTRACT DATA ===
ghi = ds_merged['ssrd']  # Surface solar radiation downward [J/m²]
temp = ds_merged['t2m']  # 2m temperature [K]
times = pd.to_datetime(ds_merged['time'].values)

# === CONVERT UNITS ===
ghi_w_m2 = ghi.values[:, 0, 0] / 3600  # J/m² to W/m²
temp_c = temp.values[:, 0, 0] - 273.15  # Kelvin to Celsius

# === COMPUTE SOLAR POWER (kW) ===
solar_power_kw = (ghi_w_m2 * panel_area * efficiency) / 1000  # in kW

# === CREATE FINAL DATAFRAME ===
df = pd.DataFrame({
    'timestamp_utc': times,
    'ghi_w_m2': ghi_w_m2,
    'temperature_c': temp_c,
    'solar_power_kw': solar_power_kw
})

df['timestamp_local'] = df['timestamp_utc'] + pd.to_timedelta(timezone_offset, unit='h')

# === SAVE TO CSV ===
df.to_csv("solar_power_2024_full.csv", index=False)
print("✅ Saved to solar_power_2024_full.csv")
