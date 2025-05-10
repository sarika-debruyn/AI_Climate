#!/usr/bin/env python3
"""
Merge solar forecasts from different models into a single file.
"""
from pathlib import Path
import sys

# Add the project root to Python's path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.shared.path_utils import solar_output

def merge_solar_forecasts():
    # Load true solar power (perfect baseline) as ground truth
    df_true = pd.read_csv(
        solar_output("solar_perfect_2023_forecast.csv"),
        dtype={"datetime": str}
    )
    
    # Convert datetime column to datetime objects
    try:
        df_true['datetime'] = pd.to_datetime(df_true['datetime'])
    except:
        # If conversion fails, assume it's the number format and create datetime objects
        df_true['datetime'] = pd.to_numeric(df_true['datetime'])  # Convert strings to numbers
        df_true['datetime'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df_true['datetime'], unit='h')
    
    df_true.set_index('datetime', inplace=True)
    df_true = df_true["power_true_MW"].rename("perfect")

    # Climatology
    df_clim = pd.read_csv(
        solar_output("solar_climatology_2023_forecast.csv"),
        dtype={"datetime": str}
    )
    
    # Convert datetime column to datetime objects
    try:
        df_clim['datetime'] = pd.to_datetime(df_clim['datetime'])
    except:
        # If conversion fails, assume it's the number format and create datetime objects
        df_clim['datetime'] = pd.to_numeric(df_clim['datetime'])  # Convert strings to numbers
        df_clim['datetime'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df_clim['datetime'], unit='h')
    
    df_clim.set_index('datetime', inplace=True)
    df_clim = df_clim["power_clim_MW"].rename("climatology")

    # NGBoost
    df_ng = pd.read_csv(
        solar_output("solar_ngboost_holdout_forecast.csv"),
        dtype={"datetime": str}
    )
    
    # Convert datetime column to datetime objects
    try:
        df_ng['datetime'] = pd.to_datetime(df_ng['datetime'])
    except:
        # If conversion fails, assume it's the number format and create datetime objects
        df_ng['datetime'] = pd.to_numeric(df_ng['datetime'])  # Convert strings to numbers
        df_ng['datetime'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df_ng['datetime'], unit='h')
    
    df_ng.set_index('datetime', inplace=True)
    df_ng = df_ng["power_pred_MW"].rename("ngboost")

    # TabPFN (convert kW to MW if needed)
    df_tpf = pd.read_csv(
        solar_output("solar_tabpfn_holdout_forecast.csv"),
        dtype={"datetime": str}
    )
    
    # Convert datetime column to datetime objects
    try:
        df_tpf['datetime'] = pd.to_datetime(df_tpf['datetime'])
    except:
        # If conversion fails, assume it's the number format and create datetime objects
        df_tpf['datetime'] = pd.to_numeric(df_tpf['datetime'])  # Convert strings to numbers
        df_tpf['datetime'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df_tpf['datetime'], unit='h')
    
    df_tpf.set_index('datetime', inplace=True)
    if "power_pred_kW" in df_tpf.columns:
        df_tpf["tabpfn"] = df_tpf["power_pred_kW"] / 1000.0
    else:
        df_tpf["tabpfn"] = df_tpf["power_pred_MW"]
    df_tpf = df_tpf[["tabpfn"]]

    # Merge all forecasts
    df_merged = pd.concat([df_true, df_clim, df_ng, df_tpf], axis=1)
    out_path = solar_output("solar_merged_forecasts.csv")
    df_merged.to_csv(out_path)
    print(f"✅ Saved merged solar forecasts to {out_path}")

if __name__ == '__main__':
    merge_solar_forecasts()
