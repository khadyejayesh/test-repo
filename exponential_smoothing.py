import pandas as pd

# Given time series data
data = [120, 130, 125, 140, 150]
alpha = 0.3  # smoothing constant

# Create DataFrame
df = pd.DataFrame({'Actual': data})

# Initialize forecast column
df['Forecast'] = 0.0

# Initial forecast (same as first actual)
df.loc[0, 'Forecast'] = df.loc[0, 'Actual']

# Apply exponential smoothing
for i in range(1, len(df)):
    df.loc[i, 'Forecast'] = (
        alpha * df.loc[i-1, 'Actual'] +
        (1 - alpha) * df.loc[i-1, 'Forecast']
    )

# Forecast for next period
next_forecast = (
    alpha * df.loc[len(df)-1, 'Actual'] +
    (1 - alpha) * df.loc[len(df)-1, 'Forecast']
)

print("Exponential Smoothing Table:\n")
print(df)

print("\nForecast for Next Period:", round(next_forecast, 2))