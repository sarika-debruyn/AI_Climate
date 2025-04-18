import pandas as pd
from pandas.api.types import CategoricalDtype

def create_climatology_model(df):
    df['year'] = df.index.year
    df['month'] = df.index.month.astype(CategoricalDtype(categories=list(range(1, 13)), ordered=True))
    df['hour'] = df.index.hour.astype(CategoricalDtype(categories=list(range(0, 24)), ordered=True))

    df_train = df[df['year'] <= 2020].copy()
    df_test = df[df['year'] >= 2021].copy()

    climatology = df_train.groupby(['month', 'hour'])['Wind Power (W)'].mean().reset_index()
    climatology.rename(columns={'Wind Power (W)': 'WindPower_climatology'}, inplace=True)

    df_test_reset = df_test.reset_index()
    df_forecast = pd.merge(df_test_reset, climatology, on=['month', 'hour'], how='left')

    return df_forecast, df_test