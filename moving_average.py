import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Sample Time Series Data (Replace with your own) ---
data = [110, 115, 120, 135, 140, 155, 160, 175, 190, 225]
years = np.arange(2015, 2025)

df = pd.DataFrame({
    "Year": years,
    "Sales": data
})

# --- Moving Average Function ---
window = 3  # change to 4, 5 etc.
df["MA_Trend"] = df["Sales"].rolling(window=window, center=True).mean()

print(df)

# --- Plotting ---
plt.figure()
plt.plot(df["Year"], df["Sales"], marker='o', label="Original Data")
plt.plot(df["Year"], df["MA_Trend"], marker='o',
         label=f"{window}-Year Moving Average", linewidth=3)

plt.xlabel("Year")
plt.ylabel("Sales")
plt.title("Trend Fitting by Moving Average Method")
plt.grid()
plt.legend()
plt.show()