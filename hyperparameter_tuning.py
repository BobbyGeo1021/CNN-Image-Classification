import tensorflow as tf  # Import TensorFlow library
from tensorflow import keras  # Import Keras from TensorFlow
from kerastuner.tuners import RandomSearch  # Import RandomSearch tuner from Keras Tuner
from cnn_model import create_cnn_model  # Import the model creation function

def build_model(hyperparameters):
    """Builds a CNN model with hyperparameters set by Keras Tuner."""
    model = create_cnn_model(input_shape=(32, 32, 3), num_classes=10)  # Create the base model
    model.compile(
        optimizer=keras.optimizers.Adam(hyperparameters['learning_rate']),  # Set learning rate from hyperparameters
        loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for multi-class classification
        metrics=['accuracy']  # Track accuracy as a metric
    )
    return model  # Return the compiled model

# Set up the hyperparameter search space
hyperparameter_space = {
    'learning_rate': [1e-2, 1e-3, 1e-4],  # Learning rates to test
    'batch_size': [32, 64, 128],  # Batch sizes to test
}

# Initialize the Keras Tuner
tuner = RandomSearch(
    build_model,  # Function to build the model
    objective='val_accuracy',  # Objective to optimize
    max_trials=10,  # Number of different models to test
    executions_per_trial=1,  # Number of executions per model trial
    directory='hyperparameter_tuning',  # Directory to save results
    project_name='cifar10_tuning'  # Project name
)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()  # Load the dataset

# Normalize the images
x_train_normalized = x_train / 255.0  # Normalize training images
x_val_normalized = x_val / 255.0  # Normalize validation images

# Start the hyperparameter search

# Use the search method with the training and validation data
# The batch size is set to a default value (e.g., 32) for the initial search
tuner.search(x_train_normalized, y_train, validation_data=(x_val_normalized, y_val), batch_size=32, epochs=10)  # Start tuning

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]  # Retrieve the best hyperparameters
print(f'Best Hyperparameters: {best_hyperparameters.values}')  # Print the best hyperparameters found
