import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
data_folder = Path("../solar-forecasting/solar_data/")  
panel_area = 1.6  # m² (typical panel)
efficiency = 0.20  # 20% conversion efficiency
timezone_offset = -7  # Yuma, AZ is in MST (UTC−7)

# === LOAD FILES ===
nc_files = sorted(data_folder.glob("*.nc"))
print(f"Found {len(nc_files)} files.")
if not nc_files:
    raise FileNotFoundError("No NetCDF files found.")

datasets = [xr.open_dataset(f) for f in nc_files]
ds = xr.concat(datasets, dim="time")
print("Available variables:", list(ds.variables))
print("ssrd shape:", ds['ssrd'].shape)

# === Select single grid point and squeeze ===
ghi = ds['ssrd'].isel(latitude=0, longitude=0).squeeze()  # J/m²
valid_time = ds['valid_time'].squeeze()

# === Flatten and convert to power ===
ghi_flat = ghi.values.flatten() / 3600  # W/m² (convert from J to W by dividing by seconds/hour)
solar_power_kw = (ghi_flat * panel_area * efficiency) / 1000  # convert W to kW

# === Handle timestamps ===
time_flat = np.tile(valid_time.values, 3)  # Assuming 3 time inits like wind
timestamps_utc = pd.to_datetime(time_flat)
timestamps_local = timestamps_utc + pd.to_timedelta(timezone_offset, unit='h')

# === Create final DataFrame ===
df = pd.DataFrame({
    'timestamp_utc': timestamps_utc,
    'timestamp_local': timestamps_local,
    'solar_power_kw': solar_power_kw
})

df = df.sort_values('timestamp_utc').reset_index(drop=True)

# === Save to CSV ===
df.to_csv("solar_power_2024.csv", index=False)
print("Saved to solar_power_2024.csv")
