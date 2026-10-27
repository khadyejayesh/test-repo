import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson

# Step 1: Create Sample Dataset
x = np.arange(1, 21)

y = [12, 15, 17, 18, 20, 23, 25, 27, 29, 30,
     33, 35, 37, 40, 42, 43, 45, 48, 50, 52]

data = pd.DataFrame({
    "Study_Hours": x,
    "Marks": y
})

print("Dataset:")
print(data.head())

# Step 2: Build Linear Regression Model
X = data[["Study_Hours"]]
Y = data["Marks"]

model = LinearRegression()
model.fit(X, Y)

prediction = model.predict(X)

# Print model parameters
print("\nIntercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Step 3: Calculate Residuals
residuals = Y - prediction

print("\nResiduals:")
print(residuals.head())

# Step 4: Durbin-Watson Test
dw = durbin_watson(residuals)

print("\nDurbin-Watson Statistic:", round(dw, 4))

# Interpretation
if dw < 1.5:
    print("Positive autocorrelation may exist.")
elif dw > 2.5:
    print("Negative autocorrelation may exist.")
else:
    print("No significant autocorrelation detected.")

# Step 5: Plot Regression Line
plt.figure(figsize=(8, 4))
plt.scatter(X["Study_Hours"], Y, color="blue", label="Actual Data")
plt.plot(X["Study_Hours"], prediction, color="red", label="Regression Line")

plt.title("Linear Regression Model")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Plot Residuals
plt.figure(figsize=(8, 4))
plt.plot(residuals, marker="o", linestyle="-", color="green")

plt.axhline(y=0, color="red", linestyle="--")

plt.title("Residual Plot")
plt.xlabel("Observation")
plt.ylabel("Residual")
plt.grid(True)

plt.show()