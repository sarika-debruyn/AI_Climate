# File: src/grid_simul/dispatch.py
#!/usr/bin/env python3

import sys
from pathlib import Path

def bootstrap_path():
    proj_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(proj_root / "src"))

bootstrap_path()

import pandas as pd
import cvxpy as cp
import numpy as np
from grid_simul.config import (
    YEAR, CAPACITY_MWH, SOC_INIT, SOC_MIN, SOC_MAX,
    RTE_CHARGE, RTE_DISCHARGE,
    P_MAX_CHARGE_MW, P_MAX_DISCHARGE_MW,
    GRID_CO2_T_PER_MWH, GRID_COST_USD_KWH
)

# Paths
DEMAND_FILE = Path(__file__).resolve().parent / f"{YEAR}_demand.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "sim_results.csv"

# Rolling-horizon lookahead window (hours)
HORIZON_HOURS = 24


def bias_correct(fc_df):
    """
    Scale ML forecasts so their mean matches 'perfect'.
    """
    mu_perfect = fc_df["perfect"].mean()
    for m in ("ngboost", "tabpfn"):
        mu_m = fc_df[m].mean()
        if mu_m > 0:
            fc_df[m] *= (mu_perfect / mu_m)
    return fc_df


def optimize_window(supply_win, demand_win, soc_init):
    """
    Solve a small LP over a lookahead window.
    Returns (c0, d0, g0, u0, soc_end).
    """
    N = len(demand_win)

    # Variables
    c   = cp.Variable(N)
    d   = cp.Variable(N)
    g   = cp.Variable(N)
    u   = cp.Variable(N)
    soc = cp.Variable(N+1)

    cons = [soc[0] == soc_init]
    for k in range(N):
        s_k = supply_win[k]
        cons += [
            soc[k+1] == soc[k] + RTE_CHARGE * c[k] - d[k] / RTE_DISCHARGE,
            c[k] >= 0, c[k] <= P_MAX_CHARGE_MW,
            d[k] >= 0, d[k] <= P_MAX_DISCHARGE_MW,
            g[k] >= 0,
            u[k] >= 0,
            s_k + d[k] + g[k] == demand_win[k] + c[k] + u[k],
            soc[k+1] >= SOC_MIN, soc[k+1] <= SOC_MAX,
        ]

    cost_per_MWh = 1000 * GRID_COST_USD_KWH
    obj = cp.Minimize(cp.sum(g) * cost_per_MWh)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except cp.SolverError:
        prob.solve(solver=cp.ECOS, verbose=False)

    # Extract first-hour decisions and final SOC
    c0 = float(c.value[0])
    d0 = float(d.value[0])
    g0 = float(g.value[0])
    u0 = float(u.value[0])
    soc_end = float(soc.value[1])
    return c0, d0, g0, u0, soc_end


def main():
    # Load demand profile
    demand = pd.read_csv(
        DEMAND_FILE, parse_dates=["datetime"], index_col="datetime"
    )["demand_MW"].values
    total_demand = demand.sum()

    results = []
    for resource in ("solar", "wind"):
        # locate forecast file
        fc_path = Path(__file__).resolve().parents[2] / "model_results" / resource / "outputs" / f"{resource}_merged_forecasts.csv"
        if not fc_path.exists():
            print(f"Missing {fc_path}, skipping {resource}")
            continue

        fc_df = pd.read_csv(fc_path, parse_dates=["datetime"], index_col="datetime")
        fc_df = bias_correct(fc_df)

        # align & clip
        fc_df = fc_df.reindex(pd.date_range(
            start=f"{YEAR}-01-01 00:00",
            end=f"{YEAR}-12-31 23:00", freq="h"
        )).fillna(0.0).clip(lower=0.0)

        for model in fc_df.columns:
            supply = fc_df[model].values
            soc = SOC_INIT
            grid_total = 0.0
            curtail_total = 0.0
            throughput_total = 0.0

            T = len(demand)
            for t in range(T):
                end = min(t + HORIZON_HOURS, T)
                sup_win = supply[t:end]
                dem_win = demand[t:end]
                c0, d0, g0, u0, soc = optimize_window(sup_win, dem_win, soc)
                grid_total      += g0
                curtail_total   += u0
                throughput_total+= (c0 + d0)

            emissions = grid_total * GRID_CO2_T_PER_MWH
            cost      = grid_total * total_demand * 0 + grid_total * 1000 * GRID_COST_USD_KWH  # grid_total MWh->kWh
            percent_met = 1 - grid_total / total_demand
            full_cycles = throughput_total / CAPACITY_MWH

            results.append({
                "resource": resource,
                "model": model,
                "grid_fallback_MWh":    round(grid_total, 2),
                "curtailment_MWh":      round(curtail_total, 2),
                "percent_demand_met":   round(percent_met, 4),
                "throughput_MWh":       round(throughput_total, 2),
                "full_equiv_cycles":    round(full_cycles, 2),
                "emissions_tCO2":       round(emissions, 2),
                "cost_USD":             round(cost, 2),
            })

    if not results:
        print("No forecast data available. Please run forecasting first.")
        return

    sim_df = pd.DataFrame(results).sort_values(["resource", "model"])
    sim_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved summary to {OUTPUT_FILE}")
    print(sim_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
