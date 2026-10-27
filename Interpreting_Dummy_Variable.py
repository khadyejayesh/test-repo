import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
data = {
    "Experience": [1, 2, 3, 4, 5, 6],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
    "Salary": [25, 28, 35, 38, 45, 48]
}

df = pd.DataFrame(data)

# Convert Gender into Dummy Variable
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

print("Dataset:")
print(df)

# Independent Variables
X = df[["Experience", "Gender"]]

# Dependent Variable
y = df["Salary"]

# Create and Train Model
model = LinearRegression()
model.fit(X, y)

# Predicted Salaries
predicted = model.predict(X)

# Display Model Details
print("\nIntercept:", model.intercept_)
print("Coefficients:")
print("Experience:", model.coef_[0])
print("Gender:", model.coef_[1])

print("\nPredicted Salaries:")
print(predicted)

# Predict Salary for a New Employee
new_employee = pd.DataFrame({
    "Experience": [7],
    "Gender": [1]   # Male = 1, Female = 0
})

new_salary = model.predict(new_employee)
print("\nPredicted Salary for New Employee:", new_salary[0])

# Plot Actual vs Predicted Salary
plt.figure(figsize=(6, 4))

plt.scatter(df["Experience"], y,
            color="blue",
            label="Actual Salary")

plt.plot(df["Experience"], predicted,
         color="red",
         label="Predicted Salary")

plt.title("Dummy Variable Regression")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()