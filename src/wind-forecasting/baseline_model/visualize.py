import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast(df_forecast):
    plt.figure(figsize=(15, 4))
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['Wind Power (W)'][-500:], label='Actual Wind Power')
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['WindPower_climatology'][-500:], label='Climatology Forecast', linestyle='--')
    plt.title('Climatology Forecast vs Actual (Last 500 Hours of Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Estimated Wind Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_forecast):
    df_heat = df_forecast[df_forecast['timestamp'].dt.year == 2022].copy()
    df_heat['abs_error'] = abs(df_heat['Wind Power (W)'] - df_heat['WindPower_climatology'])
    heatmap_data = df_heat.groupby(['month', 'hour'])['abs_error'].mean().unstack()

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5, linecolor='gray')
    plt.title("Climatology Forecast MAE by Month & Hour (2022 Only)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.show()