import pandas as pd

# Given time series data
data = [120, 130, 125, 140, 150]

# Create DataFrame
df = pd.DataFrame({'Actual': data})

# Number of periods for moving average
n = 3

# Calculate moving average forecast
df['Forecast'] = df['Actual'].rolling(window=n).mean()

# Forecast for next period
next_forecast = df['Actual'].tail(n).mean()

print("Short-Term Forecasting Table:\n")
print(df)

print("\nForecast for Next Period:", round(next_forecast, 2))