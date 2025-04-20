import pandas as pd

def train_climatology_model(df_train):
    climatology = df_train.groupby(['month', 'hour'])['GHI'].mean().reset_index()
    climatology.rename(columns={'GHI': 'GHI_climatology'}, inplace=True)
    return climatology

def apply_climatology_model(df_test, climatology):
    df_test = df_test.reset_index()
    df_forecast = pd.merge(df_test, climatology, on=['month', 'hour'], how='left')
    return df_forecast


# === File: src/evaluate_model.py ===
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(df_forecast):
    mae = mean_absolute_error(df_forecast['GHI'], df_forecast['GHI_climatology'])
    rmse = np.sqrt(mean_squared_error(df_forecast['GHI'], df_forecast['GHI_climatology']))
    return mae, rmse
