import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def climatology_forecast(df_ready):
    climatology = df_ready.groupby(['month', 'hour'])['GHI'].mean().reset_index()
    climatology.rename(columns={'GHI': 'GHI_climatology'}, inplace=True)

    df_reset = df_ready.reset_index()
    df_merged = pd.merge(df_reset, climatology, on=['month', 'hour'], how='left')

    return df_merged
