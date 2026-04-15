import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1) Create Data
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
df["Trend"] = df["Sales"].rolling(window=12, center=True).mean()

# ------------------------------
# 3) Ratio = (Actual / Trend) × 100
# ------------------------------
df["Ratio"] = (df["Sales"] / df["Trend"]) * 100

# Remove rows where Trend is NaN
clean_df = df.dropna()

# ------------------------------
# 4) Compute Seasonal Indices
# ------------------------------
seasonal_index = clean_df.groupby("Month")["Ratio"].mean()

# Normalize so average = 100
seasonal_index = (seasonal_index / seasonal_index.mean()) * 100

print("Seasonal Indices (Ratio-to-Trend Method):\n")
print(seasonal_index.round(2))

# ------------------------------
# 5) Plot Seasonal Indices
# ------------------------------
plt.figure(figsize=(10, 4))
plt.plot(seasonal_index.index, seasonal_index.values, marker='o')

plt.xlabel("Month")
plt.ylabel("Seasonal Index")
plt.title("Seasonal Indices – Ratio to Trend Method")
plt.grid(True)
plt.show()