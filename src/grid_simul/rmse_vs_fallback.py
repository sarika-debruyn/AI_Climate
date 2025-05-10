#!/usr/bin/env python3
# File: scripts/rmse_vs_fallback.py

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Bootstrap src into PYTHONPATH ───────────────────────────────
proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "src"))

from grid_simul.dispatch import greedy_dispatch
from grid_simul.config import YEAR

# ── Paths ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEMAND_FILE = proj_root / "src" / "grid_simul" / f"{YEAR}_demand.csv"
MODEL_ROOT  = proj_root / "model_results"
RMSE_DIR    = proj_root / "model_results" / "rmse"
UNCERT_FILE = proj_root / "model_results" / "sim" / "sim_results_uncertainty.csv"
OUT_PNG     = proj_root / "model_results" / "sim" / "rmse_vs_fallback.png"

# ── Load data ───────────────────────────────────────────────────
dem_df     = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"], index_col="datetime")
demand     = dem_df["demand_MW"].values
summary_df = pd.read_csv(UNCERT_FILE)

# ── Settings ────────────────────────────────────────────────────
resources = ["solar","wind"]
models    = ["ngboost","tabpfn"]

# ── Build RMSE summaries ────────────────────────────────────────
records = []
for resource in resources:
    for model in models:
        # 1) get RMSE
        # Try both possible file names and use the one that exists
        rmse_file1 = MODEL_ROOT / resource / "outputs" / f"{resource}_{model}_holdout_rmse.csv"
        rmse_file2 = MODEL_ROOT / resource / "outputs" / f"{resource}_{model}_2023_holdout_rmse.csv"
        
        if rmse_file1.exists():
            rmse_file = rmse_file1
        elif rmse_file2.exists():
            rmse_file = rmse_file2
        else:
            raise FileNotFoundError(f"Could not find RMSE file for {resource} {model}")
            
        rmse_df = pd.read_csv(rmse_file)
        if "rmse" in rmse_df.columns:
            rmse_val = rmse_df["rmse"].mean()
        else:
            rmse_val = rmse_df.iloc[:, 1].mean()

        # 2) get median fallback
        fallback = summary_df[
            (summary_df.resource == resource) &
            (summary_df.model    == model)
        ]["fallback_P50_MWh"].item()

        records.append({
            "resource": resource,
            "model":    model,
            "rmse_MW":  rmse_val,
            "fallback_MWh": fallback
        })

df = pd.DataFrame(records)

# ── Plot scatter ───────────────────────────────────────────────
plt.figure(figsize=(8,6))
colors = {"solar":"C0","wind":"C1"}

for resource in resources:
    sub = df[df.resource == resource]
    plt.scatter(sub["rmse_MW"], sub["fallback_MWh"],
                color=colors[resource], label=resource, s=100)
    for _, row in sub.iterrows():
        plt.annotate(row["model"],
                     (row["rmse_MW"], row["fallback_MWh"]),
                     textcoords="offset points", xytext=(5,5))

plt.xlabel("Hold-out RMSE (MW)")
plt.ylabel("Median grid fallback (MWh)")
plt.title("Forecast quality vs. Operational grid fallback")
plt.grid(True)
plt.legend(title="Resource")
plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────
plt.savefig(OUT_PNG)
plt.close()
print(f"Saved scatter plot to {OUT_PNG}")
