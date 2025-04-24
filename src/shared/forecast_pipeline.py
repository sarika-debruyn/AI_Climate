# === Forecast Pipeline for All Models ===
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

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

# === NGBoost Forecast ===
def forecast_ngboost(model_path, data_path, output_file, power_label):
    print(f"📈 Forecasting with NGBoost: {output_file}")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values("datetime")
    X = df.drop(columns=["datetime", power_label])
    df[power_label] = model.predict(X)
    df[["datetime", power_label]].to_csv(FORECAST_RESULTS_DIR / output_file, index=False)
    print(f"✅ Saved NGBoost forecast to forecast_results/{output_file}")

# === TabPFN Forecast ===
def forecast_tabpfn(model_path, data_path, output_file, power_label):
    print(f"🤖 Forecasting with TabPFN: {output_file}")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values("datetime")
    X = df.drop(columns=["datetime", power_label])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tab_model = TabPFNRegressor(device=device)
    tab_model.model = model
    df[power_label] = tab_model.predict(X)
    df[["datetime", power_label]].to_csv(FORECAST_RESULTS_DIR / output_file, index=False)
    print(f"✅ Saved TabPFN forecast to forecast_results/{output_file}")

# === Run All Forecasts ===
def main():
    forecast_solar_baseline()
    forecast_wind_baseline()

    forecast_ngboost(
        model_path=MODEL_RESULTS_DIR / "solar_ngboost_model.pkl",
        data_path=MODEL_RESULTS_DIR / "solar_2024_features.csv",
        output_file="solar_ngboost_forecast.csv",
        power_label="solar_power_mw"
    )
    forecast_ngboost(
        model_path=MODEL_RESULTS_DIR / "wind_ngboost_model.pkl",
        data_path=MODEL_RESULTS_DIR / "wind_2024_features.csv",
        output_file="wind_ngboost_forecast.csv",
        power_label="wind_power_mw"
    )
    forecast_tabpfn(
        model_path=MODEL_RESULTS_DIR / "solar_tabpfn_model.pkl",
        data_path=MODEL_RESULTS_DIR / "solar_2024_features.csv",
        output_file="solar_tabpfn_forecast.csv",
        power_label="solar_power_mw"
    )
    forecast_tabpfn(
        model_path=MODEL_RESULTS_DIR / "wind_tabpfn_model.pkl",
        data_path=MODEL_RESULTS_DIR / "wind_2024_features.csv",
        output_file="wind_tabpfn_forecast.csv",
        power_label="wind_power_mw"
    )

if __name__ == "__main__":
    main()
