# File: src/solar-forecasting/baseline_model/perfect.py
#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))   # add …/src
import pandas as pd
from sklearn.metrics import mean_squared_error
from shared.path_utils import SOLAR_DATA_DIR, solar_output  # creates output dirs on import

# === Solar Farm Constants ===
PANEL_AREA_TOTAL  = 256_000   # m² for 40 MW solar farm
EFFICIENCY_BASE   = 0.20      # panel efficiency
PERFORMANCE_RATIO = 0.8       # system performance ratio

TRAIN_YEARS  = range(2018, 2023)  # use 2018–2022 for consistency
HOLDOUT_YEAR = 2023

# --- Helper Functions ---

def load_solar_data(base_dir=SOLAR_DATA_DIR, years=range(2018, 2024)):
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
    mse_ghi = mean_squared_error(ghi_true, ghi_pred)
    rmse_ghi = np.sqrt(mse_ghi)

    mse_pw  = mean_squared_error(power_true, power_pred)
    rmse_pw  = np.sqrt(mse_pw)


    # Save forecasts
    out = pd.DataFrame({
        "datetime":      ghi_true.index,
        "GHI_true":      ghi_true.values,
        "GHI_pred":      ghi_pred.values,
        "power_true_MW": power_true.values,
        "power_pred_MW": power_pred.values
    }).set_index("datetime")
    out.to_csv(solar_output("solar_perfect_2023_forecast.csv"))

    # Save RMSE
    pd.DataFrame([
        {"metric": "RMSE_GHI",      "value": rmse_ghi},
        {"metric": "RMSE_power_MW", "value": rmse_pw}
    ]).to_csv(solar_output("solar_perfect_2023_rmse.csv"), index=False)

    print(" Solar perfect-foresight baseline saved:")
    print(f"   forecast → {solar_output('solar_perfect_2023_forecast.csv')}")
    print(f"   RMSE     → {solar_output('solar_perfect_2023_rmse.csv')}")

if __name__ == '__main__':
    main()
