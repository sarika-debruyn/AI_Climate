Sarika de Bruyn sed2194

AI-Driven Weather Forecasting for Renewable-Powered Data Centers: A Comparative Feasibility Study of Wind and Solar Regions

As data centers are increasing in demand, so are their carbon emissions. Therefore, the reliance of data centers on renewable energy is essential for meeting climate targets. This project explores how forecast accuracy and uncertainty impact the performance of solar- and wind-powered data centers equipped with battery storage. Using two case studies—solar energy in Arizona and wind energy in West Virginia—we simulate how machine learning-based forecasts influence energy planning, grid fallback, emissions, and economic outcomes.

We incorporate battery storage in all scenarios to reflect realistic infrastructure and allow for temporal energy shifting. The data center operates with a time-varying demand model and uses next-day forecasts of renewable availability to plan energy use. When renewables fall short, the system draws from batteries first and falls back to the grid only if needed.

We evaluate three forecasting approaches: a baseline climatology model, NGBoost (a probabilistic gradient boosting model), and TabPFN (a neural probabilistic model for tabular data). These models produce point or distributional forecasts that inform how the data center schedules energy usage and charging/discharging of storage.

Our method will use publicly available datasets from:

1. 

2. 

3. 

