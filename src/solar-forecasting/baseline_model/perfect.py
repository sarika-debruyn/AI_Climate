# --- solar_perfect_baseline.py ---
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

# === Solar Farm Constants ===
PANEL_AREA_TOTAL  = 256_000   # m² for 40 MW solar farm
EFFICIENCY_BASE   = 0.20      # panel efficiency
PERFORMANCE_RATIO = 0.8       # system performance ratio

TRAIN_YEARS  = range(2018, 2023)  # use 2018–2022 for consistency
HOLDOUT_YEAR = 2023

# --- Helper Functions ---

def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    parts = []
    for yr in years:
        path = Path(base_dir) / f"solar_{yr}.csv"
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        df = (pd.read_csv(path, skiprows=2)
              .assign(datetime=lambda d: pd.to_datetime(
                  dict(year=d.Year, month=d.Month, day=d.Day,
                       hour=d.Hour, minute=d.Minute)
              ))
              .set_index('datetime'))
        df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
        parts.append(df.dropna(subset=['GHI']))
    return pd.concat(parts).sort_index()


def ghi_to_power(ghi):
    """Convert GHI (W/m²) to farm-level power (MW)."""
    return (ghi * PANEL_AREA_TOTAL * EFFICIENCY_BASE * PERFORMANCE_RATIO) / 1e6

# --- Main Script ---

def main():
    os.makedirs("model_results", exist_ok=True)

    # Load 2018–2023 data
    df = load_solar_data()
    df_hold = df[df.index.year == HOLDOUT_YEAR]

    # Perfect forecast = actual
    ghi_true = df_hold['GHI']
    ghi_pred = ghi_true.copy()

    # Convert to power
    power_true = ghi_to_power(ghi_true)
    power_pred = ghi_to_power(ghi_pred)

    # Evaluate RMSE
    rmse_ghi = mean_squared_error(ghi_true, ghi_pred)
    rmse_pw  = mean_squared_error(power_true, power_pred)

    # Save forecasts
    out = pd.DataFrame({
        'GHI_true':      ghi_true,
        'GHI_pred':      ghi_pred,
        'power_true_MW': power_true,
        'power_pred_MW': power_pred
    })
    out.index.name = 'datetime'
    out.to_csv("model_results/solar_perfect_2023_forecast.csv")

    # Save RMSE
    pd.DataFrame([
        {'metric':'RMSE_GHI',         'value':rmse_ghi},
        {'metric':'RMSE_power_MW',    'value':rmse_pw}
    ]).to_csv("model_results/solar_perfect_2023_rmse.csv", index=False)

    print("Solar perfect-foresight baseline saved.")

if __name__ == '__main__':
    main()