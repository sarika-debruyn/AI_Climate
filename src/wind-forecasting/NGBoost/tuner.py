# Use Baysian Optimization for tuning

import optuna
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd

from main import load_wind_data, prepare_features, generate_2024_features, FEATURE_COLUMNS

# === Load and prep your data ===
df = load_wind_data()
X_train, y_train = prepare_features(df)
df_forecast = generate_2024_features()
X_forecast = df_forecast[FEATURE_COLUMNS]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_forecast_scaled = scaler.transform(X_forecast)

def objective(trial):
    # Define search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "minibatch_frac": trial.suggest_float("minibatch_frac", 0.6, 1.0),
        "natural_gradient": trial.suggest_categorical("natural_gradient", [True, False])
    }

    model = NGBRegressor(Dist=Normal, Score=MLE, verbose=False, **params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_forecast_scaled)

    # Evaluate performance
    mae = mean_absolute_error(y_pred / 1000, y_train[-len(y_pred):] / 1000)  # MW

    return mae

# === Launch optimization ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # Try 30 configurations

# === Best parameters
print("\n Best parameters found:")
print(study.best_params)
print(f" Best MAE: {study.best_value:.4f} MW")
