# AI for Climate: Solar and Wind Forecasting

This project focuses on developing machine learning models for solar and wind power forecasting, with applications in grid management and renewable energy optimization.

## Project Structure

```
AI_Climate/
├── data/                    # Raw and processed data
├── model_results/           # Model outputs and results
├── src/
│   ├── eval_forecasting/  # Evaluation scripts
│   ├── grid_simul/         # Grid simulation code
│   ├── shared/             # Shared utilities
│   ├── solar-forecasting/  # Solar forecasting models
│   └── wind-forecasting/   # Wind forecasting models
├── requirements.txt        # Python dependencies
└── Makefile               # Build automation
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
