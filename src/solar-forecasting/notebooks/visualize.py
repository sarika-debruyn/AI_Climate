import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast(df_forecast):
    plt.figure(figsize=(15, 4))
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI'][-500:], label='Actual GHI')
    plt.plot(df_forecast['timestamp'][-500:], df_forecast['GHI_climatology'][-500:], label='Climatology Forecast', linestyle='--')
    plt.title('Climatology Forecast vs Actual (Last 500 Hours)')
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_forecast, year=2022):
    df = df_forecast[df_forecast['timestamp'].dt.year == year].copy()
    df['abs_error'] = abs(df['GHI'] - df['GHI_climatology'])
    heatmap_data = df.groupby(['month', 'hour'])['abs_error'].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5, linecolor='gray')
    plt.title(f"Climatology Forecast MAE by Month & Hour ({year} Only)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.show()
