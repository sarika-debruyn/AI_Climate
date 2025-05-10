#!/usr/bin/env python3
# File: scripts/solar_cost_distribution.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Bootstrap src/ into PYTHONPATH ─────────────────────────────
proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "src"))

from grid_simul.dispatch import greedy_dispatch, optimize_calibration
from grid_simul.config import YEAR, GRID_COST_USD_KWH

# ── Settings ───────────────────────────────────────────────────
NSIMS     = 50
resource  = "solar"
models    = ["climatology", "ngboost", "tabpfn"]
cost_factor = 1000 * GRID_COST_USD_KWH  # USD per MWh

# ── Paths ───────────────────────────────────────────────────────
DEMAND_FILE = proj_root / "src" / "grid_simul" / f"{YEAR}_demand.csv"
FC_FILE     = proj_root / "model_results" / resource / "outputs" / f"{resource}_merged_forecasts.csv"
OUTPUT_PNG  = proj_root / "model_results" / "sim" / f"{resource}_cost_distribution.png"

# ── Load demand & forecasts ────────────────────────────────────
dem_df = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"], index_col="datetime")
demand = dem_df["demand_MW"].values

fc_df = pd.read_csv(FC_FILE, parse_dates=["datetime"], index_col="datetime")
fc_df = fc_df.reindex(dem_df.index).fillna(0.0)

# ── Build cost draws ────────────────────────────────────────────
cost_draws = []

# 1) Climatology (deterministic repeated)
clim = fc_df["climatology"].values
clim_m = greedy_dispatch(clim, demand)
clim_cost = clim_m["grid_fallback_MWh"] * cost_factor
cost_draws.extend([clim_cost] * NSIMS)

# 2) NGBoost & TabPFN Monte Carlo
for model in models[1:]:
    raw     = fc_df[model].values
    perfect = fc_df["perfect"].values

    # affine calibration
    a, b   = optimize_calibration(raw, perfect, demand)
    fc_cal = np.maximum(a * raw + b, 0.0)

    # residual σ
    sigma = np.std(perfect - fc_cal, ddof=0)

    # simulate
    for _ in range(NSIMS):
        synth = np.maximum(fc_cal + np.random.randn(len(fc_cal))*sigma, 0.0)
        m     = greedy_dispatch(synth, demand)
        cost_draws.append(m["grid_fallback_MWh"] * cost_factor)

# reshape into [3 × NSIMS]
data = np.array(cost_draws).reshape(len(models), NSIMS)

# ── Plot boxplot ───────────────────────────────────────────────
plt.figure(figsize=(6,5))
plt.boxplot(data.T, labels=models, notch=True, showfliers=False)
plt.title("Solar dispatch‐cost distribution")
plt.ylabel("Grid energy cost (USD)")
plt.grid(alpha=0.3)

# save
OUTPUT_PNG.parent.mkdir(exist_ok=True, parents=True)
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
plt.close()
print(f"Saved solar cost‐distribution plot to {OUTPUT_PNG}")
