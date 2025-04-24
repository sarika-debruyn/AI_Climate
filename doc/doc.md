Rory Eastland-Fruit rie2104
Sarika de Bruyn sed2194

The doc/ directory contains the LaTeX document that you are writing. We will provide a
template for your final paper.

## Assumptions

### Geographic Locations:
We need to choose the USA cities that isolate solar and wind as dominant energy sources while also ensuring that the renewable resource in each location is plentiful, consistent, and well-documented. 
1. **Solar Energy:** Yuma, Arizona, was selected as the solar-dominant site in this study due to its exceptional solar resource profile and infrastructure feasibility. According to the National Renewable Energy Laboratory (NREL), Yuma receives an average annual Global Horizontal Irradiance of over 6.5 kWh/m²/day, placing it among the highest in the continental United States [1]. The city experiences approximately 4,000 hours of sunshine per year, earning recognition by the National Climatic Data Center (NCDC) as the sunniest city in the U.S. [2]. This makes Yuma an ideal location for testing the effectiveness of machine learning-based solar forecasting models, as it offers a consistent and abundant solar resource baseline.

3. **Wind energy:** Mount Storm, West Virginia, was selected as the wind-dominant site for this study due to its strong, consistent wind resources and existing large-scale wind energy infrastructure. It is home to the Mount Storm Wind Farm, a 264 MW facility consisting of 132 turbines spread across 12 miles [3]. This makes it one of the largest wind energy installations in the eastern United States. According to the National Renewable Energy Laboratory (NREL), the region experiences average wind speeds of 7.5 to 8.5 m/s at 100 meters above ground level—well above the threshold for utility-scale wind generation [4]. Furthermore, the site is integrated into the Eastern Interconnection grid, ensuring transmission access and operational relevance. Its combination of real-world deployment, strong wind resource, and available historical weather data makes Mount Storm an ideal case study for evaluating the effectiveness of machine learning-based wind forecasting in renewable-powered data center planning.

### Data center characteristics:
We are testing for forecast variability so we need to isolate for other variables.
1. The data center has a dynamic demand. We use the following time-varying model:
   D(t) = (C + Vc X h(t) x d(t) x s(t)) x ϵ(t)
   D(t): demand at time t (MW)
   C: constant base load (MW)
   Vc: Max variable load (MW)
   h(t): Hourly profile (0-1)
   d(t): day of week adjustment
   s(t): seasonal adjustment
   ϵ(t): random noise

   * We model seasonal variation in data center demand using multipliers of $s(t) = 1.1$ for summer, $0.95$ for winter, and $1.0$ for spring and fall, based on the observed increase in cooling-related energy use reported by Shehabi et al. (8). To represent random variation in demand, we include a noise term $\epsilon(t) \sim \mathcal{N}(1.0, 0.05)$, following the forecast uncertainty range suggested by Qiu et al. (9).

   
3. Operates 24/7, 365 days a year, with no demand-side flexibility
4. All electricity is intended to be supplied by on-site renewable infrastructure 
5. When renewable output is insufficient, the center draws from the local power grid.

To model a large-scale data center powered by renewable energy, we assume a **peak power demand of 40 MW**. This assumption aligns with the infrastructure scale of regional commercial or hyperscale data centers commonly used by cloud providers. Based on average server power draw and Power Usage Effectiveness (PUE) benchmarks, we estimate the number of servers and power distribution as follows:

- **Peak Total Power**: 40 MW  
- **Power Usage Effectiveness (PUE)**: 1.33  
- **Power per Server**: 500 W (0.5 kW)
- **IT Load** = 40 MW / 1.33 ≈ **30.08 MW**
- **Estimated Servers** = 30.08 MW / 0.5 kW ≈ **60,160 servers**
- **Non-IT Load** (cooling, lighting, etc.) ≈ **10 MW**

These values reflect current trends in data center design. According to the Uptime Institute's 2024 Global Survey:
- The **average PUE across all data centers is 1.56**, with newer, more efficient builds achieving **PUEs of 1.3 or better**.
- **Server rack densities** remain mostly under 8 kW, supporting the per-server power assumption.
- Industry trends indicate increasing infrastructure densification driven by high-performance computing and business applications.

#### 📚 Source:
> Uptime Institute. (2024). *Global Data Center Survey 2024*. Uptime Intelligence Report. Available from: [https://uptimeinstitute.com](https://uptimeinstitute.com)


### Renewable Energy Infrastructure:
We need a baseline conversion from weather to power in order to assess how ML forecast quality affects power output, ensuring consistency across geographies. The data centers have a fixed number of wind turbines or solar panels sized to match peak production under optimal conditions.
1. For solar energy (Arizona):
- We assume a solar panel efficiency of 20% and a performance ratio of 0.8, values consistent with well-designed PV systems in high-irradiance regions such as Arizona [5]
- Orientation and tilt optimized for local latitude

2. For wind energy (West Virginia):
- Wind turbine efficiency = 45%
- Turbine height = 100m
- Air density = 1.225 kg/m³ (standard)
- Power output is modeled using a standard turbine power curve

### Power Conversion:
1. Solar: P = A x r x H X PR
A: panel area (normalized to meet 10 MW peak)
r: efficiency (20%)
H: solar irradiance (W/m²)
PR: performance ratio (0.8)

2. Wind: P= 0.5 x ρ × A x v^3 x n
ρ: air density (1.225 kg/m³)
A: rotor swept area
v: wind speed (m/s)
η: turbine efficiency (45%)

### Economic Conversion:
- Cost of electricity from solar: $0.04–$0.045/kWh [6]
- Cost of electricity from wind: $0.03–$0.035/kWh [6]
- Cost of grid electricity: $0.13/kWh [7]



### Sources:
1. NREL. “National Solar Radiation Data Base (NSRDB).” U.S. Department of Energy. https://nsrdb.nrel.gov
2. NOAA National Climatic Data Center. “U.S. Sunshine Rankings.” https://www.ncei.noaa.gov 
3. NREL. Wind Energy in West Virginia (Fact Sheet). Technical Report NREL/TP-6A20-70738. https://www.nrel.gov/docs/fy18osti/70738.pdf
4. NREL. U.S. Wind Resource Map at 100m. U.S. Department of Energy. https://windexchange.energy.gov/maps-data
5. NREL. PVWatts Version 5 Manual. Technical Report NREL/TP-6A20-62641. 2014. https://www.nrel.gov/docs/fy14osti/62641.pdf
6. https://www.irena.org/Publications/2024/Sep/Renewable-Power-Generation-Costs-in-2023 
7. EIA. https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_3
8. https://eta-publications.lbl.gov/sites/default/files/lbnl-1005775_v2.pdf
9. https://www.mdpi.com/1996-1073/12/4/646
10. https://umu.diva-portal.org/smash/get/diva2:957163/FULLTEXT01.pdf --> more variable sources

