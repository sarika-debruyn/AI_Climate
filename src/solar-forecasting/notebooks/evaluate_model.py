from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(df_forecast):
    mae = mean_absolute_error(df_forecast['GHI'], df_forecast['GHI_climatology'])
    rmse = np.sqrt(mean_squared_error(df_forecast['GHI'], df_forecast['GHI_climatology']))
    return mae, rmse