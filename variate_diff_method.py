import numpy as np
# Given time series data (after removing trend & seasonal effects)
data = np.array([50, 52, 49, 51, 50])
# Step 1: Calculate successive differences
differences = np.diff(data)
# Step 2: Square the differences
squared_diff = differences ** 2
# Step 3: Apply formula
n = len(data)
variance_random = np.sum(squared_diff) / (2 * (n - 1))
print("Data:", data)
print("Differences:", differences)

print("Squared Differences:", squared_diff)
print("Variance of Random Component:", variance_random)