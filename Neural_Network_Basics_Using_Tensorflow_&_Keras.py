# Import the Libraries

import tensorflow as tf
from tensorflow import keras
tf.keras.Model()
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# normalize the data : 0.255 -> 0.1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Plot the data

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()

# model

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Flattens our image to reduce to 1-D
    keras.layers.Dense(128, activation = 'relu'), # Fully connected layer
    keras.layers.Dense(10), # Final layer
])

print(model.summary())

# We can write in this from also

'''model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(10))
print(model.summary())'''

# Loss & optimizer

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # For multiclass problem because y is an int class level also sometimes label include onehot
optim = keras.optimizers.Adam(lr=0.001) # create optimizer lr is the hyper parameter
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics) # configure the model for training

# training

batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# evaluate the model

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

# predictions

probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# 2nd way
# model + softmax

predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# 3rd way

predictions = model.predict(x_test, batch_size=batch_size)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# For 5 different labels

pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis = 1)
print(label05s)

# Or we can do in another way

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)