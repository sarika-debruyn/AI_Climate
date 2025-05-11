# Project Makefile
.PHONY: all clean solar_forecast wind_forecast grid_simul

PYTHON = python3

# Main targets
all: solar_forecast wind_forecast grid_simul

# Solar Forecasting
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

solar_forecast: solar_climatology solar_perfect solar_ngboost solar_tabpfn

solar_climatology:
	$(PYTHON) src/solar-forecasting/baseline_model/climatology.py

solar_perfect:
	$(PYTHON) src/solar-forecasting/baseline_model/perfect.py

solar_ngboost:
	$(PYTHON) src/solar-forecasting/NGBoost/main.py

solar_tabpfn:
	$(PYTHON) src/solar-forecasting/tabpfn/main.py

# Wind Forecasting
wind_forecast: wind_climatology wind_perfect wind_ngboost wind_tabpfn

wind_climatology:
	$(PYTHON) src/wind-forecasting/baseline_model/climatology.py

wind_perfect:
	$(PYTHON) src/wind-forecasting/baseline_model/perfect.py

wind_ngboost:
	$(PYTHON) src/wind-forecasting/NGBoost/main.py

wind_tabpfn:
	$(PYTHON) src/wind-forecasting/tabpfn/main.py


# Evaluation
eval_solar:
	$(PYTHON) src/eval_forecasting/eval_solar.py

eval_wind:
	$(PYTHON) src/eval_forecasting/eval_wind.py

eval_rmse:
	$(PYTHON) src/eval_forecasting/eval_rmse.py

# Grid Simulation
grid_simul: rmse_vs_fallback solar_cost wind_cost

rmse_vs_fallback:
	$(PYTHON) src/grid_simul/rmse_vs_fallback.py

solar_cost:
	$(PYTHON) src/grid_simul/solar_cost_distribution.py

wind_cost:
	$(PYTHON) src/grid_simul/wind_cost_distribution.py
	
# Clean up
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.csv" -not -path "./model_results/*" -delete
