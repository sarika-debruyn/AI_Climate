import pandas as pd

def load_and_clean_data(file_paths):
    df = pd.concat([pd.read_csv(path, skiprows=2) for path in file_paths], ignore_index=True)

    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    df['Hour'] = df['Hour'].astype(int)
    df['Minute'] = df['Minute'].astype(int)

    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('timestamp', inplace=True)

    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df = df.dropna(subset=['GHI'])

    df['month'] = df.index.month
    df['hour'] = df.index.hour

    return df[['GHI', 'month', 'hour']]
