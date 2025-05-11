# AI for Climate: Solar and Wind Forecasting using ML Models for Data Centers

This project focuses on developing machine learning models (TabPFN and NGBoost) for solar and wind power forecasting for data-centers, with applications in grid management and renewable energy optimization.

## Project Structure

```
AI_Climate/
.
├── Makefile                 # Build automation file
├── README.md                # Project documentation
├── abstract.md              # Project abstract
├── doc
│   ├── Project_Paper.pdf    # Final Paper 
├── etc                      # Miscellaneous files
├── journal.md               # Notes for self-use
├── model_results            # Outputs from SRC
│   ├── rmse                 # RMSE results
│   │   ├── holdout_rmse_bar_chart.png
│   │   ├── holdout_rmse_combined.png
│   │   ├── holdout_rmse_cv_bar_chart.png
│   │   └── rmse_comparison_table.png
│   ├── sim                  # Simulation results
│   │   ├── cdf_results.csv
│   │   ├── cdf_visual.png
│   │   ├── combined_cost_distribution.png
│   │   ├── cost_vs_uncertainty.csv
│   │   ├── grid_visual.png
│   │   ├── sim_results.csv
│   │   ├── sim_results_uncertainty.csv
│   │   ├── solar_cost_distribution.png
│   │   └── wind_cost_distribution.png
│   ├── solar               # Solar model results
│   │   ├── outputs
│   │   │   ├── solar_climatology_2023_forecast.csv
│   │   │   ├── solar_climatology_2023_rmse.csv
│   │   │   ├── solar_climatology_profile.csv
│   │   │   ├── solar_merged_forecasts.csv
│   │   │   ├── solar_ngboost_best.json
│   │   │   ├── solar_ngboost_best_params.json
│   │   │   ├── solar_ngboost_cv.csv
│   │   │   ├── solar_ngboost_holdout_forecast.csv
│   │   │   ├── solar_ngboost_holdout_rmse.csv
│   │   │   ├── solar_perfect_2023_forecast.csv
│   │   │   ├── solar_perfect_2023_rmse.csv
│   │   │   ├── solar_tabpfn_cv (6).csv
│   │   │   ├── solar_tabpfn_holdout_forecast.csv
│   │   │   └── solar_tabpfn_holdout_rmse.csv
│   │   └── visuals
│   │       ├── solar_all_models_timeseries.png
│   │       └── solar_jan1-7_timeseries.png
│   └── wind               # Wind model results
│       ├── outputs
│       │   ├── wind_climatology_2023_forecast.csv
│       │   ├── wind_climatology_2023_rmse.csv
│       │   ├── wind_climatology_profile.csv
│       │   ├── wind_merged_forecasts.csv
│       │   ├── wind_ngboost_best_params.json
│       │   ├── wind_ngboost_cv.csv
│       │   ├── wind_ngboost_holdout_forecast.csv
│       │   ├── wind_ngboost_holdout_rmse.csv
│       │   ├── wind_perfect_2023_forecast.csv
│       │   ├── wind_perfect_2023_rmse.csv
│       │   ├── wind_tabpfn_2023_holdout_forecast.csv
│       │   ├── wind_tabpfn_2023_holdout_rmse.csv
│       │   └── wind_tabpfn_cv.csv
│       └── visuals
│           ├── wind_all_models_timeseries.png
│           └── wind_jan1-7_timeseries.png
├── requirements.txt      # Python dependencies
└── src                   # Source code
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-311.pyc
    ├── eval_forecasting
    │   ├── eval_rmse.py
    │   ├── eval_solar.py
    │   ├── eval_wind.py
    │   ├── merge_solar.py
    │   ├── merge_wind.py
    │   └── test.py
    ├── grid_simul         # Grid simulation
    │   ├── 2023_demand.csv
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-311.pyc
    │   │   ├── config.cpython-311.pyc
    │   │   ├── dispatch.cpython-311.pyc
    │   │   └── sim_uncertainty.cpython-311.pyc
    │   ├── cdf_visual.py
    │   ├── combined_cost_distribution.py
    │   ├── config.py
    │   ├── dispatch.py
    │   ├── generate_demand.py
    │   ├── grid_visual.py
    │   ├── sim_uncertainty.py
    │   ├── solar_cost_distribution.py
    │   └── wind_cost_distribution.py
    ├── shared
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-311.pyc
    │   │   ├── battery_simulator.cpython-311.pyc
    │   │   ├── demand_model.cpython-311.pyc
    │   │   └── path_utils.cpython-311.pyc
    │   └── path_utils.py
    ├── solar-forecasting   # Solar forecasting models
    │   ├── NGBoost
    │   │   ├── __pycache__
    │   │   │   └── main.cpython-311.pyc
    │   │   ├── feature_importance.csv
    │   │   ├── main.py
    │   │   └── solar_ngboost_best.json
    │   ├── __init__.py
    │   ├── baseline_model
    │   │   ├── README.md
    │   │   ├── __pycache__
    │   │   │   ├── climatology_model.cpython-311.pyc
    │   │   │   ├── data_loader.cpython-311.pyc
    │   │   │   ├── evaluate_model.cpython-311.pyc
    │   │   │   ├── solar_power_calc.cpython-311.pyc
    │   │   │   └── visualize.cpython-311.pyc
    │   │   ├── climatology.py
    │   │   └── perfect.py
    │   ├── solar_data
    │   │   ├── solar_2018.csv
    │   │   ├── solar_2019.csv
    │   │   ├── solar_2020.csv
    │   │   ├── solar_2021.csv
    │   │   ├── solar_2022.csv
    │   │   └── solar_2023.csv
    │   └── tabpfn
    │       ├── main.py
    │       ├── solar_TabPFN (4).ipynb
    │       └── test.py
    ├── src.txt
    └── wind-forecasting    # Wind forecasting models
        ├── NGBoost
        │   ├── __pycache__
        │   │   └── main.cpython-311.pyc
        │   ├── main.py
        │   ├── model_results
        │   ├── wind_feature_importance.csv
        │   └── wind_ngboost_best_params.json
        ├── README.md
        ├── __init__.py
        ├── baseline_model
        │   ├── README.MD
        │   ├── climatology.py
        │   └── perfect.py
        ├── tabpfn
        │   ├── main.py
        │   └── wind_TabPFN (4).ipynb
        └── wind_data
            ├── README.md
            ├── wind_2018.csv
            ├── wind_2019.csv
            ├── wind_2020.csv
            ├── wind_2021.csv
            ├── wind_2022.csv
            └── wind_2023.csv
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AI_Climate
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline

To run the entire pipeline (solar forecasting, wind forecasting, and grid simulation):

```bash
make all
```

### Individual Components

- **Solar Forecasting**:
  ```bash
  make solar_forecast
  ```

- **Wind Forecasting**:
  ```bash
  make wind_forecast
  ```

- **Grid Simulation**:
  ```bash
  make grid_simul
  ```

## Models

### Solar Forecasting
- Climatology Baseline
- Perfect Forecast
- NGBoost
- TabPFN (requires Google Colab for GPU acceleration)

### Wind Forecasting
- Climatology Baseline
- Perfect Forecast
- NGBoost
- TabPFN (requires Google Colab for GPU acceleration)

## Dependencies

- Python 3.8+
- Core: numpy, pandas, matplotlib, seaborn
- ML: scikit-learn, ngboost, torch, tabpfn, optuna
- Solar: pvlib
- Testing: pytest, pytest-cov

See `requirements.txt` for the complete list of dependencies.

## Google Colab Setup

For TabPFN models, use Google Colab with GPU acceleration:

1. Upload the project to Google Drive
2. Mount Google Drive in Colab
3. Install requirements:
   ```python
   !pip install -r requirements.txt
   ```
4. Run the TabPFN scripts:
   ```python
   !python src/solar-forecasting/tabpfn/main.py
   !python src/wind-forecasting/tabpfn/main.py
   ```

## Results

Model outputs are saved in the `model_results/` directory, including:
- Forecasts
- Evaluation metrics
- Cost distribution plots
- Grid simulation results

## License
MIT License

[Add your license here]
