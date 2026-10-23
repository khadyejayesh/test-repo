# Practical No: 3
# Aim: Implement RNN for Time Series Prediction.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

x = np.linspace(0, 100, 500)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

def create_dataset(series, window_size):
    X = []
    Y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size])
    return np.array(X), np.array(Y)

window_size = 20

X, y = create_dataset(y, window_size)

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(SimpleRNN(50, activation = 'tanh', input_shape=(window_size, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss = 'mse')
model.summary()

history = model.fit(X_train, y_train, epochs = 20, validation_split = 0.2)

y_pred = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, label = 'Actual')
plt.plot(y_pred, label = 'Predicted')
plt.title('Time Series Prediction using RNN')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
