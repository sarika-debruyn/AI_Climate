# === Demand + Battery Simulation Pipeline for All Forecast Files ===
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add src/ to path

import pandas as pd
import numpy as np
import os

from shared.demand_model import generate_time_varying_demand
from shared.battery_simulator import simulate_battery_dispatch

# === Config ===
INPUT_DIR = Path("../results")
OUTPUT_DIR = INPUT_DIR
BATTERY_CAPACITY_MWH = 20
MAX_RATE_MW = 10
EFFICIENCY = 0.9

# === List of Forecast Files ===
forecast_files = [
    "solar_ngboost_forecast.csv",
    "solar_tabpfn_forecast.csv",
    "solar_baseline_forecast.csv",
    "wind_ngboost_forecast.csv",
    "wind_tabpfn_forecast.csv",
    "wind_baseline_forecast.csv"
]

# === Load Forecast ===
def load_forecast(path):
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    else:
        raise KeyError(f"No datetime or timestamp column found in {path.name}")

    df.set_index("datetime", inplace=True)
    return df

# === Apply Simulation ===
def run_simulation_all():
    for filename in forecast_files:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename.replace("forecast", "simulated")

        print(f"\n🔄 Running simulation on: {input_path.name}")
        try:
            df = load_forecast(input_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        df = generate_time_varying_demand(df)

        if 'solar_power_mw' in df.columns:
            df.rename(columns={'solar_power_mw': 'power_mw'}, inplace=True)
        elif 'wind_power_mw' in df.columns:
            df.rename(columns={'wind_power_mw': 'power_mw'}, inplace=True)
        elif 'GHI_climatology' in df.columns:
            PANEL_AREA = 1.6
            EFFICIENCY_BASE = 0.20
            df['power_mw'] = (df['GHI_climatology'] * PANEL_AREA * EFFICIENCY_BASE) / 1000
        elif 'WindPower_climatology' in df.columns:
            df['power_mw'] = df['WindPower_climatology'] / 1e6
        else:
            print(f"⚠️ Skipping {filename}: no recognized forecast column found.")
            continue

        df['power_mw'] = df['power_mw'].fillna(0)
        df['solar_power_mw'] = df['power_mw']  # unify column name for sim function

        df = simulate_battery_dispatch(df, battery_capacity_mwh=BATTERY_CAPACITY_MWH,
                                       max_rate_mw=MAX_RATE_MW, efficiency=EFFICIENCY)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(output_path)
        print(f"Saved simulated results to: {output_path.name}")

if __name__ == "__main__":
    run_simulation_all()
