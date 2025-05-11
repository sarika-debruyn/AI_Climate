# File: scripts/grid_visual.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ── Bootstrap src/ into PYTHONPATH ──────────────────────────
proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "src"))

from grid_simul.dispatch import greedy_dispatch
from grid_simul.config import YEAR

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DEMAND_FILE = SCRIPT_DIR.parents[1] / "src" / "grid_simul" / f"{YEAR}_demand.csv"
UNCERT_FILE = SCRIPT_DIR.parents[1] / "model_results" / "sim" / "sim_results_uncertainty.csv"
MODEL_ROOT  = SCRIPT_DIR.parents[1] / "model_results"
OUTPUT_PNG  = SCRIPT_DIR.parents[1] / "model_results" / "sim" / "grid_visual.png"

# ── Load demand & summary ────────────────────────────────────
dem_df     = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"], index_col="datetime")
demand     = dem_df["demand_MW"].values
summary_df = pd.read_csv(UNCERT_FILE)

resources = ["solar", "wind"]
ml_models = ["ngboost", "tabpfn"]
models    = ["climatology"] + ml_models
x         = np.arange(len(models))
width     = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Improved Bar Plot with Data Labels

# ── Panel 1: Grid fallback ────────────────────────────────────
for i, res in enumerate(resources):
    # Climatology baseline (no MC uncertainty)
    fc_clim = pd.read_csv(
        MODEL_ROOT / res / "outputs" / f"{res}_merged_forecasts.csv",
        parse_dates=["datetime"], index_col="datetime"
    )["climatology"].reindex(dem_df.index).fillna(0.0).values
    clim_met = greedy_dispatch(fc_clim, demand)
    clim_med, clim_low, clim_high = (
        clim_met["grid_fallback_MWh"], 0.0, 0.0
    )

    # ML model summaries from your sim_uncertainty output
    sub = summary_df[summary_df.resource == res]
    ml_med = [sub[sub.model == m]["fallback_P50_MWh"].item() for m in ml_models]
    ml_low = [sub[sub.model == m]["fallback_P50_MWh"].item() - sub[sub.model == m]["fallback_P5_MWh"].item() for m in ml_models]
    ml_high = [sub[sub.model == m]["fallback_P95_MWh"].item() - sub[sub.model == m]["fallback_P50_MWh"].item() for m in ml_models]

    meds = [clim_med] + ml_med
    lows = [clim_low] + ml_low
    highs = [clim_high] + ml_high

    pos = x + (i - 0.5) * width
    bars = ax1.bar(pos, meds, width, label=res, alpha=0.7)
    ax1.errorbar(pos, meds, yerr=[lows, highs], fmt="none", capsize=4)

    # Add data labels on top of each bar
    for bar, med in zip(bars, meds):
        height = bar.get_height()
        ax1.annotate(f'{med:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel("Grid fallback (MWh)")
ax1.set_title("Grid fallback, median ± P5–P95 (based on summary statistics)")
ax1.legend()

# ── Panel 2: % Demand met ────────────────────────────────────
for i, res in enumerate(resources):
    # Climatology baseline
    fc_clim = pd.read_csv(
        MODEL_ROOT / res / "outputs" / f"{res}_merged_forecasts.csv",
        parse_dates=["datetime"], index_col="datetime"
    )["climatology"].reindex(dem_df.index).fillna(0.0).values
    clim_met = greedy_dispatch(fc_clim, demand)
    clim_med, clim_low, clim_high = (
        clim_met["percent_demand_met"], 0.0, 0.0
    )

    # ML summaries
    sub = summary_df[summary_df.resource == res]
    ml_med = [sub[sub.model == m]["met_P50_pct"].item() for m in ml_models]
    ml_low = [sub[sub.model == m]["met_P50_pct"].item() - sub[sub.model == m]["met_P5_pct"].item() for m in ml_models]
    ml_high = [sub[sub.model == m]["met_P95_pct"].item() - sub[sub.model == m]["met_P50_pct"].item() for m in ml_models]

    meds = [clim_med] + ml_med
    lows = [clim_low] + ml_low
    highs = [clim_high] + ml_high

    pos = x + (i - 0.5) * width
    bars = ax2.bar(pos, meds, width, label=res, alpha=0.7)
    ax2.errorbar(pos, meds, yerr=[lows, highs], fmt="none", capsize=4)

    # Add data labels on top of each bar
    for bar, med in zip(bars, meds):
        height = bar.get_height()
        ax2.annotate(f'{med:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel("% Demand met")
ax2.set_title("% Demand met, median ± P5–P95 (based on summary statistics)")
ax2.legend()

fig.tight_layout()
fig.savefig(OUTPUT_PNG)
plt.close(fig)
print(f"Saved enhanced grid visual with data labels to {OUTPUT_PNG}")
