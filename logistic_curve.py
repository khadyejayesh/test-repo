import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data (replace with your own) ---
np.random.seed(1)
x = np.linspace(0, 10, 50)
y = 0.5 + (5 - 0.5) / (1 + np.exp(-(x - 4))) + np.random.normal(0, 0.12, 50)

# --- Logistic Model (4-parameter) ---
def logistic4(x, A, K, B, M):
    return A + (K - A) / (1 + np.exp(-B * (x - M)))

# --- Fit ---
p0 = [min(y), max(y), 1, x[np.argmax(np.diff(y))]]  # initial guess
bounds = ([-np.inf, -np.inf, 0, min(x)], [np.inf, np.inf, np.inf, max(x)])

popt, pcov = curve_fit(logistic4, x, y, p0=p0, bounds=bounds)

A, K, B, M = popt
y_pred = logistic4(x, *popt)

# --- R² ---
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# --- Print results ---
df = pd.DataFrame({
    "Param": ["A", "K", "B", "M"],
    "Value": popt,
    "StdErr": np.sqrt(np.diag(pcov))
})

print(df)
print("R² =", r2)

# --- Plot Fit ---
plt.figure()
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, label="Logistic Fit")
plt.legend()
plt.grid()
plt.title("Logistic Curve Fit")
plt.show()

# --- Residual Plot ---
plt.figure()
plt.axhline(0, linestyle="--")
plt.scatter(x, y - y_pred)
plt.title("Residual Plot")
plt.grid()
plt.show()