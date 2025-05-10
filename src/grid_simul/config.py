# File: src/grid_simul/config.py
"""
Central configuration for demand synthesis and dispatch simulation.
"""
from pathlib import Path
import numpy as np

# ── Repo navigation ─────────────────────────────────────────
THIS_DIR    = Path(__file__).resolve().parent
ROOT_DIR    = THIS_DIR.parents[2]       # project_root/
RESULT_DIR  = ROOT_DIR / "model_results"

# ── Demand model params ─────────────────────────────────────
YEAR           = 2023        # simulation year
BASE_LOAD_MW   = 18.0        # constant base load C (MW)
VAR_CAP_MW     = 10.0        # variable capacity Vc (MW)
SEASON_FACTORS = {            # seasonal factor s(t) by month
    1: 0.95, 2: 0.95, 3: 1.00, 4: 1.00,
    5: 1.05, 6: 1.10, 7: 1.10, 8: 1.05,
    9: 1.00, 10:1.00, 11:0.95, 12:0.95
}
HOURLY_PROFILE = np.array([   # hourly profile h(t)
    0.60, 0.55, 0.50, 0.48, 0.50, 0.60,
    0.75, 0.90, 0.95, 1.00, 0.98, 0.95,
    0.95, 0.95, 0.98, 1.00, 0.95, 0.85,
    0.80, 0.78, 0.75, 0.70, 0.65, 0.60
])
EPS_MEAN       = 1.0
EPS_STD        = 0.05
RANDOM_SEED    = 42

# ── Battery & grid params ───────────────────────────────────
CAPACITY_MWH        = 40.0   # usable battery capacity (MWh)
SOC_INIT            = 0.5 * CAPACITY_MWH
SOC_MIN             = 0.1 * CAPACITY_MWH
SOC_MAX             = 0.9 * CAPACITY_MWH
RTE_CHARGE          = 0.96   # charging efficiency
RTE_DISCHARGE       = 0.96   # discharging efficiency
P_MAX_CHARGE_MW     = 10.0   # max charge power (MW)
P_MAX_DISCHARGE_MW  = 10.0   # max discharge power (MW)
GRID_CO2_T_PER_MWH  = 0.50   # tCO₂ per MWh imported
GRID_COST_USD_KWH   = 0.13   # $ per kWh imported