# === demand_model.py ===
import numpy as np
import pandas as pd

def classify_season(month):
    if month in [6, 7, 8]:
        return 'summer'
    elif month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    else:
        return 'fall'

def generate_time_varying_demand(df, base_demand=7, variable_component=3):
    """Generates time-varying data center demand using a synthetic model with hourly, weekly, seasonal variations."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday  # 0 = Monday, 6 = Sunday
    df['is_weekend'] = df['weekday'] >= 5
    df['month'] = df.index.month
    df['season'] = df['month'].apply(classify_season)

    seasonal_factors = {'summer': 1.1, 'winter': 0.95, 'spring': 1.0, 'fall': 1.0}

    def hourly_factor(hour):
        return 1.1 if 10 <= hour <= 18 else 0.9

    demand_values = []
    for idx, row in df.iterrows():
        h = hourly_factor(row['hour'])
        d = 0.9 if row['is_weekend'] else 1.0
        s = seasonal_factors[row['season']]
        epsilon = np.random.normal(1.0, 0.05)
        demand = (base_demand + variable_component * h * d * s) * epsilon
        demand_values.append(demand)

    df['demand_mw'] = demand_values
    df.drop(columns=['hour', 'weekday', 'is_weekend', 'month', 'season'], inplace=True)
    return df