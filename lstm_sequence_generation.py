# Practical No: 4
# Aim: Implement LSTM for Sequence Generation.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

x = np.linspace(0, 100, 500)
y = np.sin(x)

X, Y = [], []
for i in range(len(y) - 20):
    X.append(y[i:i+20])
    Y.append(y[i+20])
X = np.array(X).reshape(-1, 20, 1)
Y = np.array(Y)

model = Sequential()
model.add(LSTM(50, input_shape=(20, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=10, verbose=1)

seq = X[-1]
predicted = []
for _ in range(50):
    p = model.predict(seq.reshape(1, 20, 1), verbose=0)[0][0]
    predicted.append(p)
    seq = np.append(seq[1:], [[p]], axis=0)

plt.plot(predicted, label = 'Predicted')
plt.title('Future Sequence Predicted by LSTM')
plt.legend()
plt.grid(True)
plt.show()
