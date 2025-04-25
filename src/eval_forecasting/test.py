import pandas as pd

# === Load your data ===
df = pd.read_csv("wind_power_2024.csv", parse_dates=['datetime'])

# === Add month and day columns ===
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.date

# === Check missing by month ===
missing_by_month = df.groupby('month')['wind_power_mw'].apply(lambda x: x.isna().sum())
total_by_month = df.groupby('month')['wind_power_mw'].count() + missing_by_month
percent_missing_by_month = (missing_by_month / total_by_month * 100).round(2)

print("📊 Missing data by month (%):")
print(percent_missing_by_month)