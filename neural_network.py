# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# 1. Load MNIST Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 2. Preprocessing
# -----------------------------
# Normalize (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten 28x28 → 784
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# -----------------------------
# 3. Build Model (Feedforward NN)
# -----------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes (0–9)
])

# -----------------------------
# 4. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# -----------------------------
# 6. Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)

# -----------------------------
# 7. Predict Example
# -----------------------------
import numpy as np

sample = X_test[0].reshape(1, 784)
prediction = model.predict(sample)

print("Predicted digit:", np.argmax(prediction))
print("Actual digit:", y_test[0])