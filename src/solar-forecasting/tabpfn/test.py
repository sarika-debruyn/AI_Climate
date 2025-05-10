import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pvlib

# 1) Load your results
# Adjust paths as needed for your workspace
df = pd.read_csv(
    "../../../model_results/solar/solar_tabpfn_holdout_forecast.csv",
    parse_dates=["datetime"], index_col="datetime"
)

# df should have columns ['power_true_MW','power_pred_MW']
# if you saved GHI as well, load that instead:
# df_ghi = pd.read_csv("…/solar_tabpfn_holdout_forecast.csv", parse_dates=["datetime"], index_col="datetime")

# 2) Time‐series plot of power
plt.figure(figsize=(10,4))
plt.plot(df.index, df["power_true_MW"],  label="True Power",    lw=1)
plt.plot(df.index, df["power_pred_MW"],  label="TabPFN Power",  lw=1)
plt.xlabel("Date");  plt.ylabel("Power (MW)")
plt.title("Solar 2023 Hold‐out: True vs. TabPFN Power")
plt.legend(loc="upper left", frameon=False)
plt.grid(ls="--", alpha=0.5)
plt.tight_layout()
plt.show()


# 3) (Optional) also plot GHI if you saved it
# df_ghi = pd.read_csv("…", parse_dates=["datetime"], index_col="datetime")
# plt.figure(figsize=(10,4))
# plt.plot(df_ghi.index, df_ghi["GHI_true"], label="True GHI",    lw=1)
# plt.plot(df_ghi.index, df_ghi["GHI_pred"], label="TabPFN GHI", lw=1)
# plt.title("Solar 2023 Hold‐out: True vs. TabPFN GHI")
# plt.legend(); plt.grid(ls="--", alpha=0.5); plt.tight_layout(); plt.show()
