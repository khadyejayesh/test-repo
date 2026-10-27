import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Size": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 5, 5],
    "Price": [30, 35, 45, 50, 60, 65, 75, 80]
}

df = pd.DataFrame(data)

# Independent variables
X = df[["Size", "Bedrooms"]]

# Dependent variable
y = df["Price"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predictions for existing data
predicted = model.predict(X)

# Print model parameters
print("Intercept:", model.intercept_)
print("Coefficients:")
print("  Size:", model.coef_[0])
print("  Bedrooms:", model.coef_[1])

# Predict new house price
new_house = pd.DataFrame({
    "Size": [2100],
    "Bedrooms": [4]
})

prediction = model.predict(new_house)

print("Predicted Price =", prediction[0])

# Plot Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(df["Size"], y, color="blue", label="Actual Price")
plt.plot(df["Size"], predicted, color="red", label="Predicted Price")

plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price")
plt.title("Multivariate Linear Regression")
plt.legend()
plt.grid(True)

plt.show()