
### Brainstorming: Forecasting Uncertainty and Battery Storage in Solar- and Wind- Powered Data Centers

# Background Understanding of Data Centers and Grids:
## Why is grid-fall back expensive?
1. Grid electricity is more expensive, especially in regions where renewable Power Purchase Agreements (PPAs) offer fixed low rates.
a. *Question to explore*: Does Arizona or West Virginia have PPAs?
2. Grid rates can fluctuate based on demand and time-of-day (especially during peak hours)
3. If you use both renewable energy infrastructure and the grid, then you are paying for both unused renewable energy capacity and grid electrictiy
4. Utilities charge extra feeds based on peak grid usage --> "demand charges"

## How does ROI relate to data centers?
ROI = (net benefit - investment costs) / Investment costs
Forecast error --> grid fallback --> higher operating costs --> lower net benefit --> lower ROI 
Idealing want high ROI

1. Data centers invest in renewable energy infrastructure (solar panels, wind turbines, battery) assuming they'll save money over time
2. if bad forecasts lead to frequent grid fallback:
   a. they use more expensive grid electricity
   b. savings shrink, taking longer to break even
   c. ROI drops

# Methodology
## Problem:
1. Data centers are increasingly becoming necessary but are large emittors of CO2
2. Solar and wind are possible sources of energy but are intermittent and depend heavily on accurate forecasts
3. If a forecast is wrong, a data center might not have enough renewable energy, leading to costly grid fallback and higher emissions.

## Goals/Research questions:
1. How does the quality and uncertainty of renewable energy forecasts affect the performance of solar- and wind-powered data centers? Are these wind- solar- powered centers more prone to poor predictions?
2. Which is more sensitive to forecast error: solar or wind?
3. Can better forecasts reduce grid fallback, emissions, and cost?
4. How does battery storage help reduce the impacts of forecast uncertainty?

## Approach:
1. Compare two types of data centers as test beds:
  a. Arizona (solar)
  b. West Virginia (wind)

2. Simulate operations under different forecast models:
  a. Climatology (historical average) --> baseline model
  b. NGBoost (probabilistic gradient boosting)
  c. TabPFN (advanced neural model with uncertainty)

3. Training and testing
  a. Use 2018-2021 train and 2022-2023 test for baseline model
  b. Use TimeSeriesSplit for training and testing ML Models--> simulates how the model would perform in rolling time windows

5. Add a realistic time-varying demand model (hourly, daily, seasonal + noise)

6. Simulate with and without battery storage (e.g., 20–50 MWh capacity)
  a. Charge when renewables exceed demand
  b. Discharge when demand exceeds available power

7. Track performance metrics:
  a. % of demand met by renewables
  b. Grid fallback frequency and volume
  c. CO₂ emissions
  d. Economic performance (ROI)

8. Battery Storage System:
   a. Parameters: battery size (20 MWh, 50 MWh), max charge/discharge rate, efficiency (90%)
   b. Dispatch:
      i. charge when generation > demand
      ii. discharge when demand > generation
      iii. grid fallback only when battery + generation can't meet demand


## Expected Outcomes / Why It Matters
1. Show how forecast accuracy and uncertainty impact energy reliability and emissions
2. Quantify how much battery storage reduces reliance on the grid, especially when forecasts are poor
3. Compare solar vs. wind in terms of reliability, predictability, and storage needs

## Climatology Method:
- Global Horizontal Irridiance (GHI) is a good proxy for solar power availability --> the higher the GHI, the more solar energy available
- Use 
