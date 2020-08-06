# Import relevant libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Load the data (MNIST)
data = keras.datasets.fashion_mnist

# Split into train and test sets
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shrink down data so it is easier to work with
train_images = train_images/255.0
test_images = test_images/255.0

""" print(train_images[7])

# Example on how it a part of the MNIST data looks like
plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()"""

# Define the architecture for the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # In order to be able to send it in to a neuron
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax") # all neurons will add up to 1, like probability
])

# Add optimizer, loss function and metrics
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Fit model
model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)

# Predict
prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()