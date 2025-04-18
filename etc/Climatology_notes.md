# Solar - powered
convert GHI to solar power using following relationship:
Power Output=GHI×Panel Area×Efficiency×Performance Ratio


Climatology MAE (2021–2023): 26.65 W/m²
Climatology RMSE (2021–2023): 68.29 W/m²

train on 2018-2020

test on 2021-2023

# Wind - powered

convert wind speed to wind power using wind power formula:
P= 1/2*ρ*Av^3

P = power (watts)
ρ = air density (1.121 kg/m^3 for mount storm, wv)
A = swept area of turbine blades --> typical turbine radius of 40 meters --> A = pi * 40^2 = 5025.5m^2
v = wind speeed (m/s)


