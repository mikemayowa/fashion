# Import necessary libraries
library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Preprocess the data
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255

# Define the model architecture
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 64,
  validation_split = 0.2
)

# Evaluate the model on the test data
metrics <- model %>% evaluate(x_test, y_test)
print(metrics)

# Save the model for future use
save_model_hdf5(model, "fashion_mnist_cnn_model.h5")

# Load the trained model for predictions
loaded_model <- load_model_hdf5("fashion_mnist_cnn_model.h5")

# Define a function to visualize predictions
predict_and_visualize <- function(index) {
  prediction <- loaded_model %>% predict(array_reshape(x_test[index,,,], c(1, 28, 28, 1)))
  predicted_class <- which.max(prediction) - 1
  cat("Predicted class:", predicted_class, "\n")
  image(array_reshape(x_test[index,,,], c(28, 28)), col = gray.colors(256))
}

# Make predictions for two sample images
predict_and_visualize(1)
predict_and_visualize(2)
