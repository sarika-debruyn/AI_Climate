#!/usr/bin/env python3
"""
Combine solar and wind cost distribution plots into a single figure.

Saves:
  – model_results/sim/combined_cost_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Bootstrap project src
proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "src"))

from grid_simul.config import GRID_COST_USD_KWH
from grid_simul.dispatch import greedy_dispatch, optimize_calibration

# Set up paths
OUTPUT_DIR = proj_root / "model_results" / "sim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Function to Add Value Labels ─────────────────────────────────────
def add_value_labels(ax):
    """Add value labels on top of each bar."""
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')

# ─── Main Script ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) Load demand & forecasts
    dem_df = pd.read_csv(
        proj_root / "src" / "grid_simul" / f"2023_demand.csv",
        parse_dates=["datetime"], index_col="datetime"
    )
    demand = dem_df["demand_MW"].values

    # 2) Load cost distributions for both resources
    models = ["climatology", "ngboost", "tabpfn"]
    resources = ["solar", "wind"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, resource in enumerate(resources):
        # Load forecasts
        fc_df = pd.read_csv(
            proj_root / "model_results" / resource / "outputs" / f"{resource}_merged_forecasts.csv",
            parse_dates=["datetime"], index_col="datetime"
        )
        fc_df = fc_df.reindex(dem_df.index).fillna(0.0)

        # Calculate costs for each model
        cost_draws = []
        for model in models:
            raw = fc_df[model].values
            perfect = fc_df["perfect"].values
            
            # Affine calibration
            a, b = optimize_calibration(raw, perfect, demand)
            fc_cal = np.maximum(a * raw + b, 0.0)
            sigma = np.std(perfect - fc_cal, ddof=0)
            
            # Monte Carlo draws
            for _ in range(50):
                noise = np.random.randn(len(fc_cal)) * sigma
                fc_synth = np.maximum(fc_cal + noise, 0.0)
                metrics = greedy_dispatch(fc_synth, demand)
                cost = metrics["grid_fallback_MWh"] * 1000 * GRID_COST_USD_KWH
                cost_draws.append(cost)

        # Plot cost distribution
        sns.violinplot(data=cost_draws, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{resource.capitalize()} Cost Distribution')
        axes[i].set_ylabel('Cost (USD)')
        axes[i].set_xlabel('Model')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Finalize and save
    plt.tight_layout()
    output_path = OUTPUT_DIR / "combined_cost_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined cost distribution plots to {output_path}")
