import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 40)
y = 2.5 * np.exp(-0.4 * x) + 0.5 + np.random.normal(scale=0.12, size=x.size)

# Model function
def mod_exp(x, a, b, c):
    return a * np.exp(b * x) + c

# --- Fit ---
popt, pcov = curve_fit(mod_exp, x, y, p0=[1, -0.1, 0])

# Standard errors
perr = np.sqrt(np.diag(pcov))

# Predictions
y_pred = mod_exp(x, *popt)

# R² calculation
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

# Print results
print("\nFitted values (a, b, c):", np.round(popt, 5))
print("Std errors:", np.round(perr, 5))
print("R²:", round(r2, 5))

# --- Plot data + fit ---
plt.scatter(x, y, label="Data")

x_s = np.linspace(x.min(), x.max(), 400)
plt.plot(x_s, mod_exp(x_s, *popt), 'r', label="Fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Modified Exponential Fit")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot residuals ---
plt.axhline(0, linestyle='--')
plt.scatter(x, y - y_pred)

plt.title("Residual Plot")
plt.xlabel("x")
plt.ylabel("Residual")
plt.grid(True)
plt.show()