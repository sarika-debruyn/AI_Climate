import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === CONFIGURATION ===
truth_file = "wind_power_2024.csv"
model_files = {
    "Baseline": "../model_results/wind_baseline_eval_forecast.csv",
    "NGBoost": "../model_results/wind_ngboost_eval_forecast.csv",
    "TabPFN": "../model_results/wind_tabpfn_eval_forecast.csv"
}

# === LOAD TRUTH DATA ===
df_truth = pd.read_csv(truth_file, parse_dates=['datetime'])

# === LOOP THROUGH MODELS ===
for model_name, pred_file in model_files.items():
    print(f"\n📊 Evaluating {model_name} Model...")

    try:
        df_pred = pd.read_csv(pred_file, parse_dates=['datetime'])
        df = pd.merge(df_truth, df_pred, on='datetime', how='inner')

        # Extract values
        y_true = df['wind_power_mw'].values
        y_pred = df['wind_power_mw'].values

        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mbe = np.mean(y_pred - y_true)

        # Results
        print(f"MAE:  {mae:.4f} mW")
        print(f"RMSE: {rmse:.4f} mW")
        print(f"R²:   {r2:.4f}")
        print(f"MBE:  {mbe:.4f} mW")

        # Optional: save merged output
        df.to_csv(f"{model_name.lower()}_wind_eval_results.csv", index=False)

    except FileNotFoundError:
        print(f"❌ Prediction file not found: {pred_file}")
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
