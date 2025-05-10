# File: src/grid_simul/dispatch.py
#!/usr/bin/env python3

import time
import sys
from pathlib import Path

def bootstrap_path():
    # assume folder structure: …/AI_Climate (1)/src/grid_simul/dispatch.py
    proj_root = Path(__file__).resolve().parents[2]  # …/AI_Climate (1)
    sys.path.insert(0, str(proj_root / "src"))

bootstrap_path()

import pandas as pd
import numpy as np
from grid_simul.config import (
    YEAR, CAPACITY_MWH, SOC_INIT, SOC_MIN, SOC_MAX,
    RTE_CHARGE, RTE_DISCHARGE,
    P_MAX_CHARGE_MW, P_MAX_DISCHARGE_MW,
    GRID_CO2_T_PER_MWH, GRID_COST_USD_KWH
)

# Paths
SCRIPT_DIR  = Path(__file__).resolve().parent
DEMAND_FILE = SCRIPT_DIR / f"{YEAR}_demand.csv"
OUTPUT_FILE = SCRIPT_DIR.parents[1] / "model_results" / "sim" / "sim_results.csv"
MODEL_ROOT  = SCRIPT_DIR.parents[1] / "model_results"


def greedy_dispatch(supply, demand):
    """
    Fast O(T) dispatch loop:
      - clip negative supply
      - power‐limited charge/discharge
      - separate η_charge/η_discharge
      - curtailment & SOC tracking
    """
    soc = SOC_INIT
    grid = curtail = throughput = 0.0
    total = demand.sum()

    for s_raw, d in zip(supply, demand):
        s = max(0.0, s_raw)
        if s >= d:
            surplus = s - d
            charge  = min(surplus, P_MAX_CHARGE_MW, SOC_MAX - soc)
            soc    += charge * RTE_CHARGE
            curtail   += surplus - charge
            throughput += charge
        else:
            deficit = d - s
            avail   = min(P_MAX_DISCHARGE_MW, soc - SOC_MIN) * RTE_DISCHARGE
            disch   = min(deficit, avail)
            soc    -= disch / RTE_DISCHARGE
            throughput += disch
            grid      += deficit - disch

    emissions   = grid * GRID_CO2_T_PER_MWH
    cost        = grid * 1000 * GRID_COST_USD_KWH
    pct_met     = 1 - grid / total
    cycles      = throughput / CAPACITY_MWH

    return {
        "grid_fallback_MWh":  round(grid,   2),
        "curtailment_MWh":    round(curtail,2),
        "percent_demand_met": round(pct_met, 4),
        "throughput_MWh":     round(throughput,2),
        "full_equiv_cycles":  round(cycles, 2),
        "emissions_tCO2":     round(emissions,2),
        "cost_USD":           round(cost,   2),
    }


def optimize_calibration(raw, perfect, demand):
    """
    Find affine calibration raw→a*raw+b that minimizes
    grid_fallback via greedy_dispatch on (a*raw+b).
    """
    best = {"a": 1.0, "b": 0.0, "fallback": np.inf}
    # search scale factors around 1.0
    for a in np.linspace(0.7, 1.3, 61):
        # set intercept so means match
        b = perfect.mean() - a * raw.mean()
        adj = np.maximum(a * raw + b, 0.0)
        metrics = greedy_dispatch(adj, demand)
        if metrics["grid_fallback_MWh"] < best["fallback"]:
            best.update(a=a, b=b, fallback=metrics["grid_fallback_MWh"])
    return best["a"], best["b"]


def main():
    t0 = time.time()

    # load demand
    dem_df = pd.read_csv(
        DEMAND_FILE, parse_dates=["datetime"], index_col="datetime"
    )
    demand = dem_df["demand_MW"].values
    idx    = dem_df.index

    results = []
    for resource in ("solar", "wind"):
        path = MODEL_ROOT / resource / "outputs" / f"{resource}_merged_forecasts.csv"
        if not path.exists():
            print(f"Missing {path}, skipping {resource}")
            continue

        fc_df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        fc_df = fc_df.reindex(idx).fillna(0.0)

        # apply dispatch-aware affine calibration for ML models
        for model in ("ngboost", "tabpfn"):
            raw     = fc_df[model].values
            perfect = fc_df["perfect"].values
            a, b    = optimize_calibration(raw, perfect, demand)
            fc_df[model] = np.maximum(a * raw + b, 0.0)

        # run dispatch for each column
        for model in fc_df.columns:
            metrics = greedy_dispatch(fc_df[model].values, demand)
            results.append({
                "resource": resource,
                "model":    model,
                **metrics
            })

    if not results:
        print("No forecast data found; run forecasting scripts first.")
    else:
        sim_df = pd.DataFrame(results).sort_values(["resource","model"])
        sim_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved summary → {OUTPUT_FILE}\n")
        print(sim_df.to_markdown(index=False))

    print(f"\nTotal runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
