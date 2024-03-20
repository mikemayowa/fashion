# Fashion MNIST Image Classification using Convolutional Neural Network (CNN)
This repository contains Python code to train a Convolutional Neural Network (CNN) model using Keras on the Fashion MNIST dataset. The trained model is then used to classify sample images from the dataset.

## Table of Contents
Introduction
Usage
Dependencies
Instructions
Output Interpretation
Model Saving

## Introduction
Fashion MNIST is a dataset consisting of grayscale images of clothing items, each belonging to one of 10 categories. This project utilizes a CNN architecture to classify these images into their respective categories.

## Usage
Clone the repository to your local machine:

`git clone https://github.com/your_username/fashion-mnist-cnn.git`

Navigate to the project directory:

`cd fashion-mnist-cnn`

Execute the Python script:

`python fashion_mnist_cnn.py`

## Dependencies
Python 3.x
Keras
NumPy
Matplotlib

## Instructions
The fashion_mnist_cnn.py script loads the Fashion MNIST dataset, preprocesses it, defines and trains the CNN model, evaluates its performance, and saves the trained model to a file named fashion_mnist_cnn_model.h5.

Upon execution, the script also makes predictions for two sample images from the test dataset and displays the images along with their predicted classes.

## Output Interpretation
During training, the script displays the training progress including the loss and accuracy metrics for each epoch.

After training, the script prints the test accuracy achieved by the trained model.

When making predictions for sample images, the script displays the images and their predicted classes.

## Model Saving
The trained CNN model is saved as fashion_mnist_cnn_model.h5 in the project directory for future use.
