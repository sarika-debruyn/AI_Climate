from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(df_forecast):
    mae = mean_absolute_error(df_forecast['Wind Power (W)'], df_forecast['WindPower_climatology'])
    rmse = np.sqrt(mean_squared_error(df_forecast['Wind Power (W)'], df_forecast['WindPower_climatology']))
    print(f"Climatology MAE (2021–2023): {mae:.2f} W")
    print(f"Climatology RMSE (2021–2023): {rmse:.2f} W")
    return mae, rmse