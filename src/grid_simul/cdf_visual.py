#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Bootstrap project src
proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "src"))

from grid_simul.sim_uncertainty import run_monte_carlo
from grid_simul.dispatch import greedy_dispatch, optimize_calibration
from grid_simul.config import YEAR

# Paths
DEMAND_CSV = proj_root / "src" / "grid_simul" / f"{YEAR}_demand.csv"
MODEL_ROOT = proj_root / "model_results"
OUTPUT_DIR = proj_root / "model_results" / "sim"

# Load demand
dem_df = pd.read_csv(DEMAND_CSV, parse_dates=["datetime"], index_col="datetime")
demand = dem_df["demand_MW"].values

resources = ["solar", "wind"]
models    = ["ngboost", "tabpfn"]
NSIMS     = 50

plt.figure(figsize=(12,6))

for i, resource in enumerate(resources, start=1):
    ax = plt.subplot(1, 2, i)
    # climatology baseline
    fc_clim = pd.read_csv(MODEL_ROOT / resource / "outputs" / f"{resource}_merged_forecasts.csv",
                          parse_dates=["datetime"], index_col="datetime")["climatology"].reindex(dem_df.index).fillna(0.0).values
    clim_metrics = greedy_dispatch(fc_clim, demand)
    clim_fb = clim_metrics["grid_fallback_MWh"]
    ax.axvline(clim_fb, color="k", linestyle="--", label="climatology")

    for model in models:
        fc_df = pd.read_csv(MODEL_ROOT / resource / "outputs" / f"{resource}_merged_forecasts.csv",
                            parse_dates=["datetime"], index_col="datetime")
        df = fc_df.reindex(dem_df.index).fillna(0.0)
        raw     = df[model].values
        perfect = df["perfect"].values
        fallbacks, _ = run_monte_carlo(resource, model, raw, perfect, demand)
        sorted_fb = np.sort(fallbacks)
        cdf = np.arange(1, len(sorted_fb)+1) / len(sorted_fb)
        ax.step(sorted_fb, cdf, where="post", label=model)

    ax.set_title(f"{resource.capitalize()} Grid-Fallback CDF")
    ax.set_xlabel("Grid fallback (MWh)")
    ax.set_ylabel("Empirical CDF")
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cdf_visual.png")
plt.close()

# Save the CDF data to CSV
results = []
for resource in resources:
    for model in models:
        fc_df = pd.read_csv(MODEL_ROOT / resource / "outputs" / f"{resource}_merged_forecasts.csv",
                            parse_dates=["datetime"], index_col="datetime")
        df = fc_df.reindex(dem_df.index).fillna(0.0)
        raw     = df[model].values
        perfect = df["perfect"].values
        fallbacks, _ = run_monte_carlo(resource, model, raw, perfect, demand)
        sorted_fb = np.sort(fallbacks)
        cdf = np.arange(1, len(sorted_fb)+1) / len(sorted_fb)
        for fb, c in zip(sorted_fb, cdf):
            results.append({
                "resource": resource,
                "model": model,
                "grid_fallback_MWh": fb,
                "cdf": c
            })

cdf_df = pd.DataFrame(results)
cdf_df.to_csv(OUTPUT_DIR / "cdf_results.csv", index=False)
print(f"Saved CDF visualization and data to {OUTPUT_DIR}")