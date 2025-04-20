import pandas as pd
from pathlib import Path

def load_wind_data(base_dir="../wind-forecasting/wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)
    df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')  # Or whatever your wind column is
    df.dropna(subset=['Wind Speed'], inplace=True)
    return df
