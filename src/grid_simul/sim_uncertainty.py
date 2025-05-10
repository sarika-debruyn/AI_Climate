# File: src/grid_simul/sim_uncertainty.py
#!/usr/bin/env python3

import time
import sys
from pathlib import Path

def bootstrap_path():
    proj_root = Path(__file__).resolve().parents[2]   # …/AI_Climate (1)
    sys.path.insert(0, str(proj_root / "src"))

bootstrap_path()

import pandas as pd
import numpy as np
from grid_simul.config import YEAR
from grid_simul.dispatch import greedy_dispatch, optimize_calibration

# Paths
BASE_DIR    = Path(__file__).resolve().parents[2]
DEMAND_FILE = BASE_DIR / "src" / "grid_simul" / f"{YEAR}_demand.csv"
MODEL_ROOT  = BASE_DIR / "model_results"
OUTPUT_FILE = BASE_DIR / "model_results" / "sim" / "sim_results_uncertainty.csv"

NSIMS = 50  # number of Monte Carlo draws

def run_monte_carlo(resource, model, raw_fc, perfect_fc, demand):
    """
    Calibrate once, estimate residual-std, then MC simulate NSIMS draws.
    Returns arrays of grid_fallback and pct_met of length NSIMS.
    """
    # 1) find best affine a,b
    a, b = optimize_calibration(raw_fc, perfect_fc, demand)
    # 2) build calibrated forecast
    fc_cal = np.maximum(a * raw_fc + b, 0.0)
    # 3) estimate residuals and std
    resid = perfect_fc - fc_cal
    sigma = resid.std(ddof=0)
    # 4) Monte Carlo draws
    fallbacks = []
    pct_mets   = []
    for _ in range(NSIMS):
        noise = np.random.randn(len(fc_cal)) * sigma
        fc_synth = np.maximum(fc_cal + noise, 0.0)
        m = greedy_dispatch(fc_synth, demand)
        fallbacks.append(m["grid_fallback_MWh"])
        pct_mets.append(m["percent_demand_met"])
    return np.array(fallbacks), np.array(pct_mets)

def summarize(arr):
    """Return P5, P50, P95 of input array."""
    return np.percentile(arr, [5,50,95]).round(2)

def main():
    t0 = time.time()
    # load demand
    dem_df = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"], index_col="datetime")
    demand = dem_df["demand_MW"].values

    rows = []
    for resource in ("solar","wind"):
        fc_path = MODEL_ROOT / resource / "outputs" / f"{resource}_merged_forecasts.csv"
        if not fc_path.exists():
            print(f"Missing {fc_path}, skipping {resource}")
            continue

        df = pd.read_csv(fc_path, parse_dates=["datetime"], index_col="datetime")
        # align
        df = df.reindex(dem_df.index).fillna(0.0)

        for model in ("ngboost","tabpfn"):
            raw = df[model].values
            perfect = df["perfect"].values

            # run MC
            fallbacks, pct_mets = run_monte_carlo(resource, model, raw, perfect, demand)

            fb_p5, fb_p50, fb_p95 = summarize(fallbacks)
            pm_p5, pm_p50, pm_p95 = summarize(pct_mets)

            rows.append({
                "resource": resource,
                "model":    model,
                "fallback_P5_MWh":  fb_p5,
                "fallback_P50_MWh": fb_p50,
                "fallback_P95_MWh": fb_p95,
                "met_P5_pct":       pm_p5,
                "met_P50_pct":      pm_p50,
                "met_P95_pct":      pm_p95
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMonte Carlo summary saved to {OUTPUT_FILE}\n")
    print(summary_df.to_markdown(index=False))
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
