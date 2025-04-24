# === forecast.py — Generates 2024 Forecasts for Baseline, NGBoost, and TabPFN Models ===
import pandas as pd
from pathlib import Path

# === Config ===
forecast_start = "2024-01-01 00:00"
forecast_end = "2024-12-31 23:00"
forecast_timestamps = pd.date_range(forecast_start, forecast_end, freq="H")
output_dir = Path("../results")
output_dir.mkdir(parents=True, exist_ok=True)

# === Shared Constants ===
PANEL_AREA = 1.6
EFFICIENCY_BASE = 0.20
AIR_DENSITY = 1.121
ROTOR_AREA = 1.6

# === Helper to Load Climatology ===
def load_climatology(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    return df

# === Generate Baseline Forecast for Solar ===
def generate_solar_baseline():
    df = pd.DataFrame({"datetime": forecast_timestamps})
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    climatology = load_climatology("../shared/climatology/solar_climatology.csv")
    df = pd.merge(df, climatology, on=["month", "hour"], how="left")
    df["solar_power_mw"] = (df["ghi_climatology"] * PANEL_AREA * EFFICIENCY_BASE) / 1000
    df[["datetime", "solar_power_mw"]].to_csv(output_dir / "solar_baseline_forecast.csv", index=False)
    print("✅ Saved solar_baseline_forecast.csv")

# === Generate Baseline Forecast for Wind ===
def generate_wind_baseline():
    df = pd.DataFrame({"datetime": forecast_timestamps})
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    climatology = load_climatology("../shared/climatology/wind_climatology.csv")
    df = pd.merge(df, climatology, on=["month", "hour"], how="left")
    df[["datetime", "wind_power_climatology"]].rename(columns={"wind_power_climatology": "wind_power_mw"}).to_csv(
        output_dir / "wind_baseline_forecast.csv", index=False
    )
    print("✅ Saved wind_baseline_forecast.csv")

if __name__ == "__main__":
    generate_solar_baseline()
    generate_wind_baseline()
