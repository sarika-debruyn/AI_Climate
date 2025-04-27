## NGBoost method:
To generate realistic synthetic forecast features for 2024, we modeled temperature, pressure, and humidity as seasonal cycles perturbed by Gaussian noise. We fixed the random seed to ensure reproducibility across experiments. This provides plausible but non-deterministic input variability for wind speed prediction.

1. Temperature
--> Instead of a perfect smooth sine wave, add seasonal noise.
--> In real life, even if July is "hot", individual days vary a lot.

2. Pressure
--> Varies a little day-to-day around 1013 hPa.
--> Introduce small Gaussian noise.

3. Relative Humidity
--> Humidity is higher in winter and lower in summer typically.
--> Add simple seasonal modulation.

4. Dew Point
--> Depends loosely on Temperature and Humidity.

