import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 75, 85]
}

df = pd.DataFrame(data)

# Independent and Dependent variables
X = df[["Study_Hours"]]
Y = df["Marks"]

# Create and train model
model = LinearRegression()
model.fit(X, Y)

# Predict marks for 9 study hours
prediction = model.predict(pd.DataFrame({"Study_Hours": [9]}))

# Print results
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("Predicted Marks for 9 hours:", prediction[0])

# Predict all values for plotting
Y_pred = model.predict(X)

# Plot
plt.scatter(X["Study_Hours"], Y, color="blue", label="Actual Data")
plt.plot(X["Study_Hours"], Y_pred, color="red", label="Regression Line")

plt.title("Univariate Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()