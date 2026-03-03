import tensorflow as tf
import ROOT

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.save("models/firstkeras.h5")

# ROOT.TMVA.Experimental.SOFIE.RModelParser_Keras.Parse("first.h5")
