#!/usr/bin/env python3
"""
Plot hold-out RMSE and coefficient of variation for solar vs. wind.

Saves:
  – model_results/rmse/holdout_rmse_bar_chart.png
  – model_results/rmse/holdout_rmse_cv_bar_chart.png
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ─── Bootstrap so we can import shared.path_utils ──────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from path_utils import solar_output, wind_output, REPO_ROOT
# ──────────────────────────────────────────────────────────────────────

# 1) Load merged forecast tables
df_solar = pd.read_csv(
    solar_output("solar_merged_forecasts.csv"),
    parse_dates=["datetime"], index_col="datetime"
)
df_wind  = pd.read_csv(
    wind_output("wind_merged_forecasts.csv"),
    parse_dates=["datetime"], index_col="datetime"
)

models     = ["perfect", "climatology", "ngboost", "tabpfn"]
domains    = {"solar": df_solar, "wind": df_wind}

# 2) Compute RMSE and mean true power
rmse_vals   = {d: [] for d in domains}
mean_power  = {}

for d, df in domains.items():
    y_true = df["perfect"]
    mean_power[d] = y_true.mean()
    for m in models:
        rmse = np.sqrt(mean_squared_error(y_true, df[m]))
        rmse_vals[d].append(rmse)

rmse_df = pd.DataFrame(rmse_vals, index=models)

# 3) Coefficient of variation (in %)
cv_df = rmse_df.copy()
for d in domains:
    cv_df[d] = 100 * cv_df[d] / mean_power[d]

# 4) Plot RMSE
plt.figure(figsize=(8, 6))
rmse_df.plot(kind="bar", rot=0)
plt.title("Hold-out RMSE (MW)")
plt.ylabel("RMSE (MW)")
plt.tight_layout()
plt.savefig(REPO_ROOT / "model_results" / "rmse" / "holdout_rmse_bar_chart.png")
plt.close()

# 5) Plot CV
plt.figure(figsize=(8, 6))
cv_df.plot(kind="bar", rot=0)
plt.title("Hold-out RMSE Coefficient of Variation (%)")
plt.ylabel("CV (%)")
plt.tight_layout()
plt.savefig(REPO_ROOT / "model_results" / "rmse" / "holdout_rmse_cv_bar_chart.png")
plt.close()
