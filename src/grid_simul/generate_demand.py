#!/usr/bin/env python3
"""
Generate synthetic hourly demand for YEAR; save in this folder.
"""
import sys
from pathlib import Path

def bootstrap_path():
    proj_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(proj_root / "src"))

bootstrap_path()

import numpy as np
import pandas as pd
from grid_simul.config import (
    YEAR, BASE_LOAD_MW, VAR_CAP_MW, SEASON_FACTORS,
    HOURLY_PROFILE, EPS_MEAN, EPS_STD, RANDOM_SEED
)

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    idx = pd.date_range(
        start=f"{YEAR}-01-01", end=f"{YEAR}-12-31 23:00", freq="h"
    )
    reps = int(np.ceil(len(idx) / len(HOURLY_PROFILE)))
    h = np.tile(HOURLY_PROFILE, reps)[:len(idx)]
    s = idx.month.map(SEASON_FACTORS).to_numpy()
    noise = rng.normal(EPS_MEAN, EPS_STD, len(idx))
    demand = (BASE_LOAD_MW + VAR_CAP_MW * h * s) * noise
    series = pd.Series(demand.round(3), index=idx, name="demand_MW")

    out = Path(__file__).resolve().parent / f"{YEAR}_demand.csv"
    series.to_csv(out, index_label="datetime")
    print(f"Wrote demand profile to {out}")

if __name__ == "__main__":
    main()
