import pandas as pd
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

def ghi_to_power(ghi, area=1.6, efficiency=0.2):
    return ghi * area * efficiency / 1000

def load_solar_data(base_dir=Path("../solar_data"), years=range(2018, 2024)):
    file_paths = [base_dir / f"solar_{year}.csv" for year in years]
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df['DHI'] = pd.to_numeric(df['DHI'], errors='coerce')
    df['DNI'] = pd.to_numeric(df['DNI'], errors='coerce')
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df['Relative Humidity'] = pd.to_numeric(df['Relative Humidity'], errors='coerce')
    df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')
    df['Cloud Type'] = pd.to_numeric(df['Cloud Type'], errors='coerce')
    df.dropna(subset=['GHI'], inplace=True)
    return df

def prepare_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['zenith_norm'] = df['zenith'] / 90.0
    df['is_clear'] = (df['Cloud Type'] == 0).astype(int)

    features = df[['sin_hour', 'cos_hour', 'dayofyear', 'zenith_norm',
                   'DHI', 'DNI', 'Temperature', 'Relative Humidity', 'Wind Speed', 'is_clear']]
    target = ghi_to_power(df['GHI'])
    return features, target

def get_feature_importance(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    importances = np.zeros(X.shape[1])

    for train_idx, val_idx in tscv.split(X):
        model = NGBRegressor(Dist=Normal, Score=MLE, verbose=False)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        importances += model.feature_importances_

    return pd.Series(importances / 5, index=X.columns).sort_values(ascending=False)

df = load_solar_data()
X, y = prepare_features(df)
importances = get_feature_importance(X, y)
print(importances)
