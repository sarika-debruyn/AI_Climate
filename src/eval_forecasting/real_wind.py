import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
data_folder = Path("../wind-forecasting/wind_data/")
air_density = 1.225  # kg/m³
rotor_radius = 40  # meters (80m rotor diameter)
rotor_area = np.pi * rotor_radius**2
timezone_offset = -5  # Mount Storm, WV = Eastern Time

# === LOAD FILES ===
nc_files = sorted(data_folder.glob("*.nc"))
print(f"Found {len(nc_files)} files.")
if not nc_files:
    raise FileNotFoundError("No NetCDF files found.")

datasets = [xr.open_dataset(f) for f in nc_files]
ds = xr.concat(datasets, dim="time")
print("Available variables:", list(ds.variables))
print("u100 shape:", ds['u100'].shape)

# === Select single grid point ===
u = ds['u100'].isel(latitude=0, longitude=0).squeeze()  # shape: (3, 8784)
v = ds['v100'].isel(latitude=0, longitude=0).squeeze()
valid_time = ds['valid_time'].squeeze()


# === Flatten everything together
u_flat = u.values.flatten()
v_flat = v.values.flatten()
time_flat = np.tile(valid_time.values, 3)  # Match the shape of u/v

# === Compute wind speed & power
wind_speed = np.sqrt(u_flat**2 + v_flat**2)
wind_power_kw = 0.5 * air_density * rotor_area * wind_speed**3 / 1000

# === Handle timestamps
timestamps_utc = pd.to_datetime(time_flat)
timestamps_local = timestamps_utc + pd.to_timedelta(timezone_offset, unit='h')

# === Create final DataFrame
df = pd.DataFrame({
    'timestamp_utc': timestamps_utc,
    'timestamp_local': timestamps_local,
    'wind_power_kw': wind_power_kw
})

# === Sort and save
df = df.sort_values('timestamp_utc').reset_index(drop=True)
df.to_csv("wind_power_2024.csv", index=False)
print("Saved to wind_power_2024.csv")