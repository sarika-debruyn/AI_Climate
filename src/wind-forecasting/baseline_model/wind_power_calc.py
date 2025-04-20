from config import AIR_DENSITY, SWEEP_AREA

def estimate_wind_power(df):
    coeff = 0.5 * AIR_DENSITY * SWEEP_AREA
    df['Wind Power (W)'] = coeff * (df['Wind Speed'] ** 3)
    return df