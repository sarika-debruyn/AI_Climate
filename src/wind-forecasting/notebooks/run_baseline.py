from data_loader import load_wind_data
from wind_power_calc import estimate_wind_power
from climatology_model import create_climatology_model
from evaluate_model import evaluate_model
from visualize import plot_forecast, plot_heatmap

if __name__ == "__main__":
    df = load_wind_data()
    df = estimate_wind_power(df)
    df_forecast, df_test = create_climatology_model(df)
    evaluate_model(df_forecast)
    plot_forecast(df_forecast)
    plot_heatmap(df_forecast)