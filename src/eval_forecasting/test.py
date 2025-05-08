import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("../../model_results/solar/solar_merged_forecasts.csv",
                 parse_dates=["datetime"], index_col="datetime")

# filter to daylight hours
day = df[df["perfect"] > 0]

print("Daylight-only metrics:")
for model in ["climatology","ngboost","tabpfn"]:
    y_true = day["perfect"]
    y_pred = day[model]
    print(f"{model:12s}"
          f"  RMSE = {mean_squared_error(y_true,y_pred):5.2f} MW"
          f"  MAE = {mean_absolute_error(y_true,y_pred):5.2f} MW")