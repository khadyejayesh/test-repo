import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    "Income": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "Expense": [18, 21, 28, 30, 39, 41, 52, 55, 69, 80]
}

df = pd.DataFrame(data)

# Independent Variable
X = df[["Income"]]

# Dependent Variable
y = df["Expense"]

# Create and Train Model
model = LinearRegression()
model.fit(X, y)

# Model Parameters
print("Intercept:", model.intercept_)
print("Slope (Coefficient):", model.coef_[0])

# Predict new value
new_income = pd.DataFrame({"Income": [70]})
prediction = model.predict(new_income)
print("Predicted Expense for Income = 70:", prediction[0])

# Predicted values for existing data
predicted = model.predict(X)

# Model Accuracy
r2 = r2_score(y, predicted)
print("R² Score:", r2)

# Residuals
residuals = y - predicted

# ------------------------------
# Plot 1: Regression Line
# ------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(X["Income"], y, color="blue", label="Actual Data")
plt.plot(X["Income"], predicted, color="red", label="Regression Line")

plt.title("Linear Regression Model")
plt.xlabel("Income")
plt.ylabel("Expense")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Plot 2: Residual Plot
# ------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(predicted, residuals, color="green")

plt.axhline(y=0, color="red", linestyle="--")

plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# Interpretation
print("\nResidual Plot Interpretation:")
print("- Random scatter around zero → No heteroscedasticity.")
print("- Funnel/Cone shape → Heteroscedasticity may exist.")