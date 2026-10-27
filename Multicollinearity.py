import pandas as pd
import matplotlib.pyplot as plt

# Sample Data
data = {
    "Size": [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Rooms": [2, 2, 3, 3, 4, 4, 5],
    "Price": [30, 35, 45, 50, 60, 65, 75]
}

df = pd.DataFrame(data)

# Correlation Matrix
corr = df.corr()

print("Correlation Matrix")
print(corr)

# Plot Correlation Matrix
plt.figure(figsize=(6, 5))

plt.imshow(corr, cmap="Blues")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)

# Display correlation values
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()