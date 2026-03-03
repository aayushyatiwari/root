import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(8)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')

x = np.random.randn(2, 32)
y = np.random.randn(2, 8)

model.fit(x, y, epochs=500, verbose=0)
model.save('models/KerasModel.keras')

