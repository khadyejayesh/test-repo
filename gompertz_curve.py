import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Data (demo random data) ---
np.random.seed(0)
x = np.linspace(0, 10, 40)
y = 10 * np.exp(-2.5 * np.exp(-0.6 * x)) + np.random.normal(0, 0.4, size=40)

# --- Gompertz model ---
def gompertz(x, A, B, C):
    return A * np.exp(-B * np.exp(-C * x))

# --- Fit curve ---
p0 = [max(y) * 1.05, 1, 0.5]  # initial guesses
popt, pcov = curve_fit(gompertz, x, y, p0=p0)

A, B, C = popt
y_pred = gompertz(x, *popt)

# --- Goodness of fit (R²) ---
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# --- Print parameters ---
df = pd.DataFrame({
    "param": ["A", "B", "C"],
    "estimate": popt,
    "std_err": np.sqrt(np.diag(pcov))
})

print(df)
print("R² =", r2)

# --- Plot Fit ---
plt.figure()
plt.scatter(x, y, label="Data")
x_s = np.linspace(x.min(), x.max(), 400)
plt.plot(x_s, gompertz(x_s, *popt), label="Gompertz Fit")
plt.legend()
plt.grid()
plt.title("Gompertz Curve Fit")
plt.show()

# --- Residual Plot ---
plt.figure()
plt.axhline(0, linestyle="--")
plt.scatter(x, y - y_pred)
plt.title("Residual Plot")
plt.grid()
plt.show()