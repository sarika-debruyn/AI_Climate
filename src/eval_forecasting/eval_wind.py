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
print(f" Loaded truth file with columns: {df_truth.columns.tolist()}")

# === LOOP THROUGH MODELS ===
for model_name, pred_file in model_files.items():
    print(f"\n Evaluating {model_name} Model...")

    try:
        df_pred = pd.read_csv(pred_file, parse_dates=['datetime'])
        # Check for NaNs in raw files before merging
        nan_truth = df_truth['wind_power_mw'].isna().sum()
        nan_pred = df_pred['wind_power_mw'].isna().sum()
        print(f"NaNs before merge — Truth: {nan_truth}, Prediction: {nan_pred}")
        # Merge on datetime
        df = pd.merge(df_truth, df_pred, on='datetime', how='inner')

        # Rename columns for clarity
        df = df.rename(columns={
            'wind_power_mw_x': 'wind_power_mw_truth',
            'wind_power_mw_y': 'wind_power_mw_pred'
        })

        if 'wind_power_mw_truth' not in df.columns or 'wind_power_mw_pred' not in df.columns:
            raise KeyError("Expected columns 'wind_power_mw_truth' and 'wind_power_mw_pred' not found after merge.")

        # Extract true and predicted values
        y_true = df['wind_power_mw_truth'].values
        y_pred = df['wind_power_mw_pred'].values

        # Compute metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mbe = np.mean(y_pred - y_true)

        # Print results
        print(f"MAE:  {mae:.4f} MW")
        print(f"RMSE: {rmse:.4f} MW")
        print(f"R²:   {r2:.4f}")
        print(f"MBE:  {mbe:.4f} MW")

        # Save merged results
        df.to_csv(f"{model_name.lower()}_wind_eval_results.csv", index=False)
        print(f"Saved merged results to: {model_name.lower()}_wind_eval_results.csv")

    except FileNotFoundError:
        print(f"Prediction file not found: {pred_file}")
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
