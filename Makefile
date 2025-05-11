# Project Makefile
.PHONY: all clean install install-dev test lint format solar_forecast wind_forecast grid_simul grid_utils baseline_models

PYTHON = python3
COLAB_PYTHON = python3  # Change this to your Colab Python path if needed
PIP = pip

# Main targets
all: baseline_models solar_forecast wind_forecast grid_simul grid_utils

# Dependency Management
install:
	@echo "Installing project dependencies..."
	$(PIP) install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 mypy pytest pytest-cov

# Development
format:
	@echo "Formatting code..."
	black src/
	isort src/

lint:
	@echo "Linting code..."
	flake8 src/
	mypy src/

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing

# Baseline Models
baseline_models: solar_baselines wind_baselines

solar_baselines: solar_climatology solar_perfect

wind_baselines: wind_climatology wind_perfect

# Solar Forecasting
solar_forecast: solar_ngboost solar_tabpfn

solar_climatology:
	$(PYTHON) src/solar-forecasting/baseline_model/climatology.py

solar_perfect:
	$(PYTHON) src/solar-forecasting/baseline_model/perfect.py

solar_ngboost:
	$(PYTHON) src/solar-forecasting/NGBoost/main.py

solar_tabpfn_colab:
	@echo "Running TabPFN model in Google Colab..."
	@echo "Please run the following commands in Google Colab:"
	@echo "1. Upload the project folder to Google Drive"
	@echo "2. Mount Google Drive"
	@echo "3. Navigate to the project directory"
	@echo "4. Run: !pip install -r requirements.txt"
	@echo "5. Run: !python src/solar-forecasting/tabpfn/main.py"

solar_tabpfn: solar_tabpfn_colab

# Wind Forecasting
wind_forecast: wind_ngboost wind_tabpfn

wind_climatology:
	$(PYTHON) src/wind-forecasting/baseline_model/climatology.py

wind_perfect:
	$(PYTHON) src/wind-forecasting/baseline_model/perfect.py

wind_ngboost:
	$(PYTHON) src/wind-forecasting/NGBoost/main.py

wind_tabpfn_colab:
	@echo "Running TabPFN model in Google Colab..."
	@echo "Please run the following commands in Google Colab:"
	@echo "1. Upload the project folder to Google Drive"
	@echo "2. Mount Google Drive"
	@echo "3. Navigate to the project directory"
	@echo "4. Run: !pip install -r requirements.txt"
	@echo "5. Run: !python src/wind-forecasting/tabpfn/main.py"

wind_tabpfn: wind_tabpfn_colab

# Evaluation
eval_solar:
	$(PYTHON) src/eval_forecasting/eval_solar.py

eval_wind:
	$(PYTHON) src/eval_forecasting/eval_wind.py

eval_rmse:
	$(PYTHON) src/eval_forecasting/eval_rmse.py

# Grid Utilities
grid_utils: config generate_demand dispatch

config:
	@echo "Configuration file is ready to use at src/grid_simul/config.py"

# Generate demand data
generate_demand:
	$(PYTHON) src/grid_simul/generate_demand.py

# Dispatch simulation
dispatch:
	$(PYTHON) src/grid_simul/dispatch.py

# Grid Simulation
grid_simul: grid_utils solar_cost wind_cost combined_cost

solar_cost:
	$(PYTHON) src/grid_simul/solar_cost_distribution.py

wind_cost:
	$(PYTHON) src/grid_simul/wind_cost_distribution.py

combined_cost:
	$(PYTHON) src/grid_simul/combined_cost_distribution.py

# Clean up
clean:
	@echo "Cleaning up..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.csv" -not -path "./model_results/*" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".mypy_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -exec rm -rf {} +
