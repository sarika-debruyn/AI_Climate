import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error


# === Wind Farm Constants ===
AIR_DENSITY    = 1.121    # kg/m³
TURBINE_RADIUS = 50       # m
SWEEP_AREA     = np.pi * TURBINE_RADIUS**2
TURBINE_COUNT  = 16       # turbines per farm
EFFICIENCY     = 0.40     # turbine efficiency

TRAIN_YEARS  = range(2018, 2023)
HOLDOUT_YEAR = 2023

# --- Helper Functions ---

def load_wind_data(base_dir="../wind_data", years=range(2018, 2024)):
    parts = []
    for yr in years:
        path = Path(base_dir) / f"wind_{yr}.csv"
        if not path.exists():
            sys.exit(f"Missing file: {path}")
        df = (pd.read_csv(path, skiprows=2)
              .assign(datetime=lambda d: pd.to_datetime(
                  dict(year=d.Year, month=d.Month, day=d.Day,
                       hour=d.Hour, minute=d.Minute)
              ))
              .set_index('datetime'))
        df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')
        parts.append(df.dropna(subset=['Wind Speed']))
    return pd.concat(parts).sort_index()


def wind_speed_to_power(ws):
    """Convert wind speed (m/s) to farm-level power (MW)."""
    coef = 0.5 * AIR_DENSITY * SWEEP_AREA * EFFICIENCY * TURBINE_COUNT
    return coef * (ws**3) / 1e6

# --- Main Script ---

def main():
    os.makedirs("../../model_results", exist_ok=True)

    # Load 2018–2023 data
    df = load_wind_data()
    df_hold = df[df.index.year == HOLDOUT_YEAR]

    # Perfect forecast = actual
    ws_true = df_hold['Wind Speed']
    ws_pred = ws_true.copy()

    # Convert to power
    power_true = wind_speed_to_power(ws_true)
    power_pred = wind_speed_to_power(ws_pred)

    # Evaluate RMSE
    rmse_ws = mean_squared_error(ws_true, ws_pred)
    rmse_pw = mean_squared_error(power_true, power_pred)

    # Save forecasts
    out = pd.DataFrame({
        'WS_true_m_s':    ws_true,
        'WS_pred_m_s':    ws_pred,
        'power_true_MW':  power_true,
        'power_pred_MW':  power_pred
    })
    out.index.name = 'datetime'
    out.to_csv("../../model_results/wind_perfect_2023_forecast.csv")

    # Save RMSE
    pd.DataFrame([
        {'metric':'RMSE_WS_m_s',    'value':rmse_ws},
        {'metric':'RMSE_power_MW',  'value':rmse_pw}
    ]).to_csv("../../model_results/wind_perfect_2023_rmse.csv", index=False)

    print("Wind perfect-foresight baseline saved.")

if __name__ == '__main__':
    main()
