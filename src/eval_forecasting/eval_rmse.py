#!/usr/bin/env python3
"""
Plot hold-out RMSE and coefficient of variation for solar vs. wind.

Saves:
  – model_results/holdout_rmse_bar_chart.png
  – model_results/holdout_rmse_cv_bar_chart.png
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

# 4) Plot raw RMSE bar chart
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, rmse_df["solar"], width, label="Solar")
ax.bar(x + width/2, rmse_df["wind"],  width, label="Wind")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("RMSE (MW)")
ax.set_title("Hold-out RMSE by Model and Domain")
ax.legend()
fig.tight_layout()

out1 = REPO_ROOT / "model_results" / "holdout_rmse_bar_chart.png"
fig.savefig(out1, dpi=300)
plt.close(fig)
print(f"Saved bar chart of RMSE to {out1}")

# 5) Plot CV bar chart
fig, ax = plt.subplots()
ax.bar(x - width/2, cv_df["solar"], width, label="Solar")
ax.bar(x + width/2, cv_df["wind"],  width, label="Wind")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Coefficient of Variation (%)")
ax.set_title("Hold-out RMSE Coefficient of Variation by Model and Domain")
ax.legend()
fig.tight_layout()

out2 = REPO_ROOT / "model_results" / "holdout_rmse_cv_bar_chart.png"
fig.savefig(out2, dpi=300)
plt.close(fig)
print(f"Saved CV bar chart to {out2}")
