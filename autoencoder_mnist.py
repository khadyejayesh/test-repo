# Practical No: 6
# Aim: Implement Auto-Encoder with tensorflow.

import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

(xtr,_),(xte,_) = tf.keras.datasets.mnist.load_data()
xtr, xte = xtr.reshape(-1,784)/255., xte.reshape(-1,784)/255.

class AE(Model):
    def __init__(self):
        super().__init__()
        self.e = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')])
        self.d = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid')])
    def call(self, x): return self.d(self.e(x))

ae = AE()
ae.compile('adam', 'mse')
ae.fit(xtr, xtr, epochs=10, batch_size=256, validation_data=(xte, xte))

out = ae.predict(xte)

plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(xte[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.subplot(2,5,i+6)
    plt.imshow(out[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()
