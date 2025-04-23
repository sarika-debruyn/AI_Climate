# === NGBoost Wind Forecast with Forecast Output and Datetime Index ===
import pandas as pd
import numpy as np
from pathlib import Path
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.model_selection import TimeSeriesSplit

# === Constants ===
AIR_DENSITY = 1.121  # kg/m³
ROTOR_AREA = 1.6     # m²

# === Step 1: Load wind data ===
def load_wind_data(base_dir="../wind-forecasting/wind_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"wind_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    for col in ['Wind Speed', 'Temperature', 'Relative Humidity', 'Pressure', 'Cloud Type', 'Dew Point']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Wind Speed'], inplace=True)
    return df.set_index('datetime').sort_index()

# === Step 2: Wind power conversion (kW) ===
def wind_speed_to_power(wind_speed, rho=AIR_DENSITY, area=ROTOR_AREA):
    return 0.5 * rho * area * (wind_speed ** 3) / 1000  # output in kW

# === Step 3: Feature engineering ===
def prepare_features(df):
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['sin_hour', 'cos_hour', 'dayofyear',
                   'Temperature', 'Pressure', 'Relative Humidity', 'Dew Point', 'is_clear']]
    target = wind_speed_to_power(df['Wind Speed'])
    return features, target

# === Step 4: Train, evaluate and forecast ===
def evaluate_model(X, y, timestamps, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    all_predictions = []

    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = NGBRegressor(Dist=Normal, Score=MLE, verbose=False)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_pred - y_val))
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))

        maes.append(mae)
        rmses.append(rmse)

        preds = pd.DataFrame({
            'datetime': timestamps.iloc[val_idx].values,
            'wind_power_mw': y_pred / 1000  # convert to MW
        })
        all_predictions.append(preds)

        print(f"Fold {i+1}: MAE = {mae:.4f} | RMSE = {rmse:.4f}")

    print(f"\nAverage MAE: {np.mean(maes):.4f}")
    print(f"Average RMSE: {np.mean(rmses):.4f}")

    return pd.concat(all_predictions).sort_values(by='datetime').reset_index(drop=True)

# === Main ===
def main():
    print("Loading wind data...")
    df = load_wind_data()

    print("Preparing features...")
    X, y = prepare_features(df)
    timestamps = df.index.to_series()

    print("Evaluating model performance and generating forecasts...")
    forecast_df = evaluate_model(X, y, timestamps)

    print("Saving forecast results...")
    forecast_df.to_csv("results/wind_ngboost_forecast.csv", index=False)

if __name__ == "__main__":
    main()
