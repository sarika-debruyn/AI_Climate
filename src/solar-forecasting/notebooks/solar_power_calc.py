def add_power_output(df, panel_area_m2=1000, efficiency=0.20, performance_ratio=0.8):
    df = df.copy()
    df['solar_output_W'] = df['GHI'] * panel_area_m2 * efficiency * performance_ratio
    df['solar_output_kW'] = df['solar_output_W'] / 1000
    return df