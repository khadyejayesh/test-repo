import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

# Load dataset
digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)

# Define different SVM models with varying hyperparameters
models = {
    "Underfitting (C=0.01, gamma=0.001)": SVC(kernel="rbf", C=0.01, gamma=0.001),
    "Well-balanced (C=10, gamma=0.01)": SVC(kernel="rbf", C=10, gamma=0.01),
    "Overfitting (C=1000, gamma=1)": SVC(kernel="rbf", C=1000, gamma=1),
}

# Train each model and evaluate performance
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    results[name] = (train_acc, test_acc)

# Print results
print("\nSVM Model Performance:")
for name, (train_acc, test_acc) in results.items():
    print(f"{name}: Training Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

# Plot results
labels = list(results.keys())
train_scores = [results[label][0] for label in labels]
test_scores = [results[label][1] for label in labels]

x = np.arange(len(labels))
plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, train_scores, 0.4, label='Train Accuracy', color='blue')
plt.bar(x + 0.2, test_scores, 0.4, label='Test Accuracy', color='red')
plt.xticks(x, labels, rotation=15)
plt.ylabel("Accuracy")
plt.title("Overfitting & Underfitting in SVM")
plt.legend()

plt.show()
