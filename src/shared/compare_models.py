import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def evaluate_models(baseline_path, ngboost_path, tabpfn_path, title, save_name):
    # Load model outputs
    baseline_df = pd.read_csv(baseline_path)
    ngboost_df = pd.read_csv(ngboost_path)
    tabpfn_df = pd.read_csv(tabpfn_path)

    # Parse datetime
    baseline_df['datetime'] = pd.to_datetime(baseline_df['datetime'])
    ngboost_df['datetime'] = pd.to_datetime(ngboost_df['datetime'])
    tabpfn_df['datetime'] = pd.to_datetime(tabpfn_df['datetime'])

    # Merge forecasts
    comparison_df = baseline_df[['datetime', baseline_df.columns[-1]]].rename(columns={baseline_df.columns[-1]: 'baseline_power'})
    comparison_df = comparison_df.merge(ngboost_df.rename(columns={ngboost_df.columns[-1]: 'ngboost_power'}), on='datetime', how='left')
    comparison_df = comparison_df.merge(tabpfn_df.rename(columns={tabpfn_df.columns[-1]: 'tabpfn_power'}), on='datetime', how='left')

    # Assume baseline_power is "truth" for now
    truth = comparison_df['baseline_power']

    # Calculate MAEs
    mae_baseline = mean_absolute_error(truth, comparison_df['baseline_power'])  # Should be ~0
    mae_ngboost = mean_absolute_error(truth, comparison_df['ngboost_power'])
    mae_tabpfn = mean_absolute_error(truth, comparison_df['tabpfn_power'])

    print(f"\n=== {title} ===")
    print(f"Baseline MAE: {mae_baseline:.4f} MW (trivial, self-prediction)")
    print(f"NGBoost MAE: {mae_ngboost:.4f} MW")
    print(f"TabPFN MAE: {mae_tabpfn:.4f} MW")

    # Plotting
    models = ['Baseline', 'NGBoost', 'TabPFN']
    maes = [mae_baseline, mae_ngboost, mae_tabpfn]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, maes)
    plt.title(f"Model MAE Comparison - {title}")
    plt.ylabel("Mean Absolute Error (MW)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"../model_results/{save_name}.png")  # Save the plot
    plt.show()

def main():
    print("\n===== Evaluating Wind Forecasts =====")
    evaluate_models(
        baseline_path="../model_results/wind_baseline_eval_forecast.csv",
        ngboost_path="../model_results/wind_ngboost_eval_forecast.csv",
        tabpfn_path="../model_results/wind_tabpfn_eval_forecast.csv",
        title="Wind Forecasts",
        save_name="wind_model_comparison"
    )

    print("\n===== Evaluating Solar Forecasts =====")
    evaluate_models(
        baseline_path="../model_results/solar_baseline_eval_forecast.csv",
        ngboost_path="../model_results/solar_ngboost_eval_forecast.csv",
        tabpfn_path="../model_results/solar_tabpfn_eval_forecast.csv",
        title="Solar Forecasts",
        save_name="solar_model_comparison"
    )

if __name__ == "__main__":
    main()
