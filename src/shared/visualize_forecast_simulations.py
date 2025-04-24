# === Forecast and Simulation Visualization for All Models ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

sns.set(style="whitegrid")

# === Paths ===
RESULTS_DIR = Path("../results")
forecast_files = {
    "solar_ngboost": "solar_ngboost_forecast.csv",
    "solar_tabpfn": "solar_tabpfn_forecast.csv",
    "solar_baseline": "solar_baseline_forecast.csv",
    "wind_ngboost": "wind_ngboost_forecast.csv",
    "wind_tabpfn": "wind_tabpfn_forecast.csv",
    "wind_baseline": "wind_baseline_forecast.csv"
}
simulated_files = {
    key: val.replace("forecast", "simulated")
    for key, val in forecast_files.items()
}

# === Forecast Comparison (Power Prediction Accuracy) ===
def plot_forecast_comparison(model_key, days=7):
    forecast_path = RESULTS_DIR / forecast_files[model_key]
    df = pd.read_csv(forecast_path, parse_dates=True)
    datetime_col = next(col for col in df.columns if "datetime" in col.lower())
    df["datetime"] = pd.to_datetime(df[datetime_col])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()

    if "solar" in model_key:
        df["actual_power_mw"] = (df.get("GHI", 0) * 1.6 * 0.20) / 1000
        pred_col = next(c for c in df.columns if "solar" in c or "GHI" in c or "power" in c and "climatology" not in c)
    else:
        df["actual_power_mw"] = df.get("Wind Power (W)", 0) / 1e6
        pred_col = next(c for c in df.columns if "wind" in c or "power" in c and "climatology" not in c)

    df = df[["actual_power_mw", pred_col]].dropna()
    df = df.iloc[:24*days]  # limit to days of hourly data

    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df["actual_power_mw"], label="Actual", linewidth=2)
    plt.plot(df.index, df[pred_col], label="Predicted", linestyle="--")
    plt.title(f"Forecast Accuracy - {model_key.replace('_', ' ').title()} (First {days} Days)")
    plt.ylabel("Power (MW)")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Simulation Dynamics ===
def plot_simulation_dynamics(model_key, days=7):
    sim_path = RESULTS_DIR / simulated_files[model_key]
    df = pd.read_csv(sim_path, parse_dates=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index().iloc[:24*days]

    # Handle alternative fallback and battery column naming (baseline only)
    if "grid_fallback" in df.columns:
        df.rename(columns={"grid_fallback": "grid_fallback_mw"}, inplace=True)
    if "battery_flow" in df.columns:
        df.rename(columns={"battery_flow": "battery_mw"}, inplace=True)

    required_cols = ["demand_mw", "power_mw", "grid_fallback_mw", "battery_soc"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️ Skipping simulation dynamics for {model_key}: missing columns {missing}")
        return

    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df["demand_mw"], label="Demand", color="black", linewidth=2)
    plt.plot(df.index, df["power_mw"], label="Renewable Supply", alpha=0.8)
    if "battery_mw" in df.columns:
        plt.plot(df.index, df["battery_mw"], label="Battery Flow", alpha=0.8)
    plt.plot(df.index, df["grid_fallback_mw"], label="Grid Fallback", linestyle=":")
    plt.title(f"Simulation Dynamics - {model_key.replace('_', ' ').title()} (First {days} Days)")
    plt.ylabel("Power (MW)")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot battery SOC separately
    plt.figure(figsize=(14, 3))
    plt.plot(df.index, df["battery_soc"], label="Battery State of Charge", color="green")
    plt.title(f"Battery SOC - {model_key.replace('_', ' ').title()} (First {days} Days)")
    plt.ylabel("MWh")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

# === Batch Visualization ===
def run_all_visualizations():
    for key in forecast_files:
        print(f"\n📈 Visualizing: {key}")
        plot_forecast_comparison(key)
        plot_simulation_dynamics(key)

if __name__ == "__main__":
    run_all_visualizations()
