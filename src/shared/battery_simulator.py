def simulate_battery_dispatch(df, battery_capacity_mwh=20, max_rate_mw=10, efficiency=0.9):
    """Simulates battery storage operations over time given demand and generation."""
    df = df.copy()
    df['battery_soc'] = 0.0
    df['grid_fallback'] = 0.0
    df['renewables_used'] = 0.0

    soc = 0.0
    for i in range(len(df)):
        gen = df.iloc[i]['solar_power_mw']
        demand = df.iloc[i]['demand_mw']
        net = gen - demand

        if net > 0:
            # Charge the battery
            charge = min(net, max_rate_mw, battery_capacity_mwh - soc)
            soc += charge * efficiency
            df.at[df.index[i], 'renewables_used'] = demand
        else:
            # Discharge the battery
            required = -net
            discharge = min(required, max_rate_mw, soc)
            soc -= discharge
            met_demand = gen + discharge
            fallback = max(0, demand - met_demand)

            df.at[df.index[i], 'renewables_used'] = demand - fallback
            df.at[df.index[i], 'grid_fallback'] = fallback

        df.at[df.index[i], 'battery_soc'] = soc

    return df