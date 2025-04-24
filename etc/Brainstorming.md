
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
  b. Use TimeSeriesSplit for NGBoost--> simulates how the model would perform in rolling time windows
  c. TabPFN is pretrained with neural networks via meta-learning so it doesn't need hyperparameter tuning

5. Add a realistic time-varying demand model (hourly, daily, seasonal + noise)

\begin{table}[ht]
\centering
\caption{Values for Time-Varying Demand Model Variables}
\begin{tabular}{|l|l|l|p{6cm}|}
\hline
\textbf{Variable} & \textbf{Value / Range} & \textbf{Description} & \textbf{Source / Justification} \\
\hline
$C$ & 7 MW & Constant base demand & TA recommendation; aligns with average baseline load in enterprise-scale data centers. \\
\hline
$V_c$ & 3 MW & Max variable demand component & TA recommendation; captures dynamic usage on top of base load. \\
\hline
$h(t)$ & 0.85–1.2 & Hourly variation multiplier & Based on CPU load patterns in Birke et al.~\cite{birke2012}; lower at night, peaks midday. \\
\hline
$d(t)$ & 0.9 (weekends), 1.0 (weekdays) & Daily multiplier for weekend/weekday load & Weekend demand ~10\% lower than weekday per usage profiling in Birke et al.~\cite{birke2012}. \\
\hline
$s(t)$ & 0.95 (winter), 1.0 (spring/fall), 1.1 (summer) & Seasonal multiplier for cooling impact & Increased summer HVAC load per Shehabi et al.~\cite{shehabi2016}. \\
\hline
$\epsilon(t)$ & $\mathcal{N}(1.0, 0.05)$ & Random noise term (Gaussian) & Reflects 5\% forecast error margin, consistent with workload prediction uncertainty in Mashayekhy et al.~\cite{mashayekhy2015}. \\
\hline
\end{tabular}
\label{tab:time_varying_demand}
\end{table}


7. Simulate with battery storage (e.g., 20–50 MWh capacity)
  a. Charge when renewables exceed demand
  b. Discharge when demand exceeds available power

8. Track performance metrics:
  a. % of demand met by renewables
  b. Grid fallback frequency and volume
  c. CO₂ emissions
  d. Economic performance (ROI)

9. Battery Storage System:
   a. Parameters: battery size (20 MWh, 50 MWh), max charge/discharge rate, efficiency (90%)
   b. Dispatch:
      i. charge when generation > demand
      ii. discharge when demand > generation
      iii. grid fallback only when battery + generation can't meet demand


## Expected Outcomes / Why It Matters
1. Show how forecast accuracy and uncertainty impact energy reliability and emissions
2. Quantify how much battery storage reduces reliance on the grid, especially when forecasts are poor
3. Compare solar vs. wind in terms of reliability, predictability, and storage needs

## Files and Purpose:
1. main.py --> Train + evaluate climatology model on 2021–2023
2. forecast.py -->	Generate 2024 forecast for solar and wind baseline using saved climatology
3. simulate.py -->	Run battery + grid fallback simulations using any forecast
4. visualize.py -->	Plot performance of forecasts + simulations
