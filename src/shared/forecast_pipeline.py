# === Forecast Pipeline for All Models ===
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# === Paths ===
MODEL_RESULTS_DIR = Path("../model_results")
FORECAST_RESULTS_DIR = Path("../forecast_results")
FORECAST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Generate Future Forecast Timestamps ===
def generate_forecast_timestamps(start="2024-01-01", end="2024-12-31 23:00"):
    return pd.date_range(start=start, end=end, freq="H")

# === Solar Baseline Forecast ===
def forecast_solar_baseline():
    print("🌞 Forecasting: Solar Baseline")
    climatology = pd.read_csv(MODEL_RESULTS_DIR / "solar_climatology.csv")
    timestamps = generate_forecast_timestamps()
    df = pd.DataFrame({"datetime": timestamps})
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    df = df.merge(climatology, on=["month", "hour"], how="left")
    PANEL_AREA = 1.6
    EFFICIENCY = 0.20
    df["solar_power_mw"] = (df["GHI_climatology"] * PANEL_AREA * EFFICIENCY) / 1000
    df = df[["datetime", "solar_power_mw"]].dropna()

    df.to_csv(FORECAST_RESULTS_DIR / "solar_baseline_forecast.csv", index=False)
    print("✅ Saved to forecast_results/solar_baseline_forecast.csv")

# === Wind Baseline Forecast ===
def forecast_wind_baseline():
    print("🌬️ Forecasting: Wind Baseline")
    climatology = pd.read_csv(MODEL_RESULTS_DIR / "wind_climatology.csv")
    timestamps = generate_forecast_timestamps()
    df = pd.DataFrame({"datetime": timestamps})
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    df = df.merge(climatology, on=["month", "hour"], how="left")
    df["wind_power_mw"] = df["WindPower_climatology"] / 1e6
    df = df[["datetime", "wind_power_mw"]].dropna()

    df.to_csv(FORECAST_RESULTS_DIR / "wind_baseline_forecast.csv", index=False)
    print("✅ Saved to forecast_results/wind_baseline_forecast.csv")

# === Run All Forecasts ===
def main():
    forecast_solar_baseline()
    forecast_wind_baseline()
    # TODO: Add NGBoost and TabPFN model forecasts later

if __name__ == "__main__":
    main()
