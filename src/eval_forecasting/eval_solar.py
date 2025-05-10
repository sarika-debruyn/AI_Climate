import os
import pandas as pd
import matplotlib.pyplot as plt

"""
This script produces two visualizations for solar forecasts:
1. A time-series plot of true power vs. all forecasts.
2. A boxplot of forecast errors for each model.

Outputs are saved under ../../model_results/visualizations/solar/.
"""
#------------
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

#--------------
def evaluate_solar():
    # Paths
    merged_csv = "../../model_results/solar/solar_merged_forecasts.csv"
    out_dir    = "../../model_results/visualizations/solar"
    os.makedirs(out_dir, exist_ok=True)

    # Load merged forecasts
    df = pd.read_csv(
        merged_csv,
        parse_dates=["datetime"],
        index_col="datetime"
    )

    # --- 1. Hourly metrics ---
    print("\nHourly point‐forecast metrics (Power in MW):")
    y_true = df["perfect"]
    for model in ["perfect", "climatology", "ngboost", "tabpfn"]:
        y_pred = df[model]
        rmse = mean_squared_error(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        print(f"  {model:12s}  RMSE = {rmse:6.3f}   MAE = {mae:6.3f}")

    # If you really want to see them numerically:
    errors = df[["perfect","climatology","ngboost","tabpfn"]].subtract(y_true, axis=0)
    print("\nError summary:")
    print(errors.describe())

    # --- 2. Thin‐down time‐series plot (first week of January) ---
    week = df["2023-01-01":"2023-01-07"]
    plt.figure(figsize=(10,4))
    plt.plot(week.index, week["perfect"],       label="Perfect",      ls="--", alpha=0.8)
    plt.plot(week.index, week["climatology"],   label="Climatology",  ls=":",  alpha=0.8)
    plt.plot(week.index, week["ngboost"],       label="NGBoost",      ls="-.", alpha=0.8)
    plt.plot(week.index, week["tabpfn"],        label="TabPFN",       ls="-",  alpha=0.8)
    plt.title("Solar 2023: True vs Forecasts (Jan 1–7 Hourly Detail)")
    plt.ylabel("Power (MW)")
    plt.xlabel("Date")
    plt.legend(ncol=2)
    plt.grid(ls=":", lw=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("../../model_results/visualizations/solar/solar_jan1-7_timeseries.png", dpi=300)
    plt.close()
    print("Saved detailed hourly plot for Jan 1–7 to solar_jan1-7_timeseries.png")

    # --- 1. Time-series plot ---
    # Down-sample to daily means to improve readability
    df_daily = df.resample("D").mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_daily.index, df_daily["perfect"],
            label="Perfect Forecast", color="tab:blue", linestyle="--", alpha=0.8)
    ax.plot(df_daily.index, df_daily["climatology"],
            label="Climatology", color="tab:orange", linestyle=":", alpha=0.8)
    ax.plot(df_daily.index, df_daily["ngboost"],
            label="NGBoost", color="tab:green", linestyle="-.", alpha=0.8)
    ax.plot(df_daily.index, df_daily["tabpfn"],
            label="TabPFN", color="tab:red", linestyle="-", alpha=0.8)

    ax.set_title("Solar 2023: True Power & All Forecasts (Daily Mean)")
    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Date")
    ax.grid(which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/solar_all_models_timeseries.png", dpi=300)
    plt.close(fig)
    print(f"Saved time-series plot to {out_dir}/solar_all_models_timeseries.png")

    # --- 2. Error distribution boxplot ---
    errors = df[["perfect", "climatology", "ngboost", "tabpfn"]].subtract(
        df["perfect"], axis=0
    )

    plt.figure(figsize=(6, 4))
    errors.boxplot()
    plt.title("Solar Forecast Error Distribution (2023)")
    plt.ylabel("Error (MW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/solar_error_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved error boxplot to {out_dir}/solar_error_boxplot.png")


if __name__ == '__main__':
    evaluate_solar()