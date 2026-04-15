import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1) Data (3 Years)
# ------------------------------
data = {
    "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] * 3,
    "Year": [2019]*12 + [2020]*12 + [2021]*12,
    "Sales": [
        120,130,150,160,180,200,210,220,190,170,160,140,
        130,135,155,165,190,210,220,230,200,175,165,150,
        125,140,160,170,195,215,225,235,205,180,170,155
    ]
}

df = pd.DataFrame(data)

# ------------------------------
# 2) 12-Month Moving Average (Trend)
# ------------------------------
df["MA_12"] = df["Sales"].rolling(window=12).mean()

# Centered Moving Average (CMA)
df["CMA"] = df["MA_12"].rolling(window=2).mean()

# ------------------------------
# 3) Ratio = (Actual / CMA) × 100
# ------------------------------
df["Ratio"] = (df["Sales"] / df["CMA"]) * 100

# Remove rows with NaN
clean = df.dropna()

# ------------------------------
# 4) Seasonal Index Calculation
# ------------------------------
seasonal_index = clean.groupby("Month")["Ratio"].mean()

# Normalize so average = 100
seasonal_index = (seasonal_index / seasonal_index.mean()) * 100

# Maintain correct month order
month_order = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

seasonal_index = seasonal_index.reindex(month_order)

print("\nSeasonal Indices (Ratio of Moving Average Method):\n")
print(seasonal_index.round(2))

# ------------------------------
# 5) Plot Seasonal Indices
# ------------------------------
plt.figure(figsize=(10, 4))
plt.plot(seasonal_index.index, seasonal_index.values, marker='o')

plt.title("Seasonal Indices – Ratio of Moving Average Method")
plt.xlabel("Month")
plt.ylabel("Seasonal Index")
plt.grid(True)
plt.show()