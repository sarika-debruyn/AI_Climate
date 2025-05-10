"""
Centralised, portable paths for the AI_Climate repo.
"""

from pathlib import Path
import os

# ── locate repo root (   .../repo_root/  ) ─────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # …/src/shared
SRC_DIR    = SCRIPT_DIR.parent                        # …/src
REPO_ROOT  = SRC_DIR.parent                           # repo root

# ── raw data folders ───────────────────────────────────────────────────
SOLAR_DATA_DIR = Path(os.getenv(
    "SOLAR_DATA_DIR",
    SRC_DIR / "solar-forecasting" / "solar_data"
))
WIND_DATA_DIR = Path(os.getenv(
    "WIND_DATA_DIR",
    SRC_DIR / "wind-forecasting" / "wind_data"
))

# ── results & visual folders ───────────────────────────────────────────
SOLAR_OUT_DIR  = REPO_ROOT / "model_results" / "solar" / "outputs"
WIND_OUT_DIR   = REPO_ROOT / "model_results" / "wind"  / "outputs"
SOLAR_VIS_DIR  = REPO_ROOT / "model_results" / "solar" / "visuals"
WIND_VIS_DIR   = REPO_ROOT / "model_results" / "wind"  / "visuals"

def _ensure_dirs():
    """Create output/visual directories if they don’t exist."""
    for p in (SOLAR_OUT_DIR, WIND_OUT_DIR, SOLAR_VIS_DIR, WIND_VIS_DIR):
        p.mkdir(parents=True, exist_ok=True)

# Convenience helpers ---------------------------------------------------
def solar_output(name: str): return SOLAR_OUT_DIR / name
def wind_output(name: str):  return WIND_OUT_DIR  / name
def solar_visual(name: str): return SOLAR_VIS_DIR / name
def wind_visual(name: str):  return WIND_VIS_DIR  / name

_ensure_dirs()        # run immediately on import
