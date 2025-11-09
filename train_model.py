import os  # Import os module for environment variable access
import numpy as np  # Import NumPy for numerical operations
import tensorflow as tf  # Import TensorFlow library
import logging  # Import logging module for error handling
from data_preprocessing import load_and_preprocess_image, normalize_images, create_data_generator  # Import preprocessing functions
from cnn_model import create_cnn_model, train_model  # Import model creation and training functions

# Load sensitive information from environment variables
API_KEY = os.getenv('API_KEY')  # Load API key from environment variable

# Set up logging configuration to log errors to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')  # Configure logging

try:
    # Load the CIFAR-10 dataset using the data generator
    train_generator = create_data_generator('path/to/train_data', batch_size=32)  # Create a data generator for training data
    validation_generator = create_data_generator('path/to/val_data', batch_size=32)  # Create a data generator for validation data
except FileNotFoundError as e:
    logging.error(f'FileNotFoundError: {e}')  # Log the error if the dataset cannot be found
    raise  # Re-raise the exception to exit the program

# Define input shape and number of classes
input_shape = (32, 32, 3)  # Input shape for CIFAR-10 images
num_classes = 10  # Number of classes in CIFAR-10 dataset

# Create the CNN model
cnn_model = create_cnn_model(input_shape, num_classes)  # Create the CNN model using the defined function

# Define a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  # Reduce learning rate on plateau

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop training if val_loss doesn't improve

try:
    # Train the model using the train_model function with the generator
    trained_model = train_model(cnn_model, train_generator, validation_generator, epochs=10, callbacks=[lr_scheduler, early_stopping])  # Train the model
except tf.errors.ResourceExhaustedError as e:
    logging.error(f'MemoryError during training: {e}')  # Log memory errors
    raise  # Re-raise the exception to exit the program

# Evaluate the model on the validation set
try:
    val_loss, val_accuracy = trained_model.evaluate(validation_generator)  # Evaluate the model and get loss and accuracy
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')  # Print the evaluation metrics
except Exception as e:
    logging.error(f'Error during evaluation: {e}')  # Log any other errors during evaluation
    raise  # Re-raise the exception to exit the program
