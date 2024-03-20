## Step 1: Implementing a CNN using Keras in Python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", test_accuracy)

# Save the model for future use
model.save("fashion_mnist_cnn_model.h5")




## Step 2: Making Predictions for Sample Images

# Load the trained model
from keras.models import load_model
model = load_model("fashion_mnist_cnn_model.h5")

# Define a function to visualize predictions
def predict_and_visualize(index):
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.axis('off')
    prediction = np.argmax(model.predict(X_test[index].reshape(1, 28, 28, 1)))
    print("Predicted class:", prediction)

# Make predictions for two sample images
predict_and_visualize(0)
predict_and_visualize(1)

