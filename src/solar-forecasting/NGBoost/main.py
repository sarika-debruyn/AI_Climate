import pandas as pd
import numpy as np
from pathlib import Path
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.model_selection import TimeSeriesSplit
import pvlib

# Constants 
LATITUDE = 32.7
LONGITUDE = -114.63
TIMEZONE = 'Etc/GMT+7'
PANEL_AREA = 1.6  # m²
EFFICIENCY_BASE = 0.20
TEMP_COEFF = 0.004  # 0.4% loss per °C above 25°C
T_REF = 25  

# Step 1: Load solar data 
def load_solar_data(base_dir="../solar_data", years=range(2018, 2024)):
    file_paths = [Path(base_dir) / f"solar_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    
    for col in ['GHI', 'DHI', 'DNI', 'Temperature', 'Wind Speed', 'Relative Humidity', 'Cloud Type']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['GHI'], inplace=True)
    return df

# Step 2: Add solar zenith 
def add_zenith_angle(df, latitude=LATITUDE, longitude=LONGITUDE, timezone=TIMEZONE):
    df['datetime'] = pd.to_datetime(df['datetime'])
    solar_position = pvlib.solarposition.get_solarposition(
        time=df['datetime'], latitude=latitude, longitude=longitude
    )
    df['zenith'] = solar_position['zenith'].values
    return df

# Step 3: Solar Power Estimation
def temp_derated_efficiency(temp, base_eff=EFFICIENCY_BASE, gamma=TEMP_COEFF, T_ref=T_REF):
    return base_eff * (1 - gamma * (temp - T_ref))

def ghi_to_power(ghi, temp, area=PANEL_AREA):
    eff = temp_derated_efficiency(temp)
    return ghi * area * eff / 1000  # returns power in kW

# Step 4: Feature engineering 
def prepare_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['zenith_norm'] = df['zenith'] / 90.0
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['sin_hour', 'cos_hour', 'dayofyear', 'zenith_norm',
                   'DHI', 'DNI', 'Temperature', 'Relative Humidity', 'Wind Speed', 'is_clear']]
    target = ghi_to_power(df['GHI'], df['Temperature'])
    return features, target

# Step 5: Train and evaluate model 
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
            'solar_power_mw': y_pred / 1000
        })
        all_predictions.append(preds)

        print(f"Fold {i+1}: MAE = {mae:.4f} | RMSE = {rmse:.4f}")

    print(f"\nAverage MAE: {np.mean(maes):.4f}")
    print(f"Average RMSE: {np.mean(rmses):.4f}")

    return pd.concat(all_predictions).sort_values(by='datetime').reset_index(drop=True)

# === Main ===
def main():
    print("Loading data...")
    df = load_solar_data()

    print("Calculating solar zenith...")
    df = add_zenith_angle(df)

    print("Preparing features...")
    X, y = prepare_features(df)
    timestamps = df.index.to_series()

    print("Evaluating model performance and generating forecasts...")
    forecast_df = evaluate_model(X, y, timestamps)

    print("Saving forecast results...")
    forecast_df.to_csv("../../model_results/solar_ngboost_eval_forecast.csv", index=False)

    print("Training final model on all data...")
    final_model = NGBRegressor(Dist=Normal, Score=MLE, verbose=False)
    final_model.fit(X, y)

    print("Saving final model...")
    import joblib
    joblib.dump(final_model, "../../model_results/solar_ngboost_model.pkl")
    print("✅ Saved NGBoost model to model_results/solar_ngboost_model.pkl")

    print("Saving input features for 2024...")
    df_features = X.copy()
    df_features["datetime"] = timestamps.values
    df_features["solar_power_mw"] = y.values / 1000
    df_features.to_csv("../../model_results/solar_2024_features.csv", index=False)
    print("✅ Saved 2024 features to model_results/solar_2024_features.csv")


if __name__ == "__main__":
    main()
