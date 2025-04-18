import pandas as pd
from pathlib import Path

def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df.dropna(subset=['GHI'], inplace=True)
    return df