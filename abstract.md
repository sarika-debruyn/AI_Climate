Rory Eastland-Fruit rie2104
Sarika de Bruyn sed2194

Dynamic Data Center Workload Scheduling Using Real-Time Weather Data and Machine Learning

As data centers are increasing in demand, so are their carbon emissions. In 2024, data centers were responsible for about 2% of global electricity consumption, with a significant portion of their energy usage attributed to computing workloads and cooling systems. Integrating renewable energy sources into data center operations can significantly reduce carbon emissions, but the intermittency of solar and wind energy makes it challenging to rely on renewables without effective scheduling. To address this, we propose an approach that leverages real-time weather data and machine learning-based forecasting to dynamically schedule data center workloads in alignment with periods of high renewable energy availability.
Our method will use publicly available datasets from:
NOAA’s National Centers for Environmental Information – Provides real-time weather data (solar radiation, wind speed, temperature, humidity) to predict renewable energy generation potential.
Electricity Maps API – Offers real-time grid carbon intensity data, allowing for optimization of workload scheduling based on grid emissions levels.
Google DeepMind’s AI for Data Center Energy Optimization – Historical insights into data center energy consumption trends and cooling requirements.
National Renewable Energy Laboratory (NREL) Wind & Solar Forecasting Data – Used to train time-series forecasting models for predicting renewable energy availability.

Using time-series forecasting models, we predict short-term fluctuations in renewable energy availability. We will then integrate a reinforcement learning-based optimization model to dynamically adjust data center computing workloads in real time. This ensures that energy-intensive processes run when solar and wind energy are most available, reducing reliance on fossil-fuel-powered grids.

Our approach demonstrates the potential of machine learning in improving the sustainability of data center operations by reducing emissions and optimizing energy consumption without compromising computational efficiency.

