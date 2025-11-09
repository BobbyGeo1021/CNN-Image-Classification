import tensorflow as tf  # Import TensorFlow library
from data_preprocessing import load_and_preprocess_image, normalize_images, augment_images  # Import preprocessing functions

# Define a function to create the CNN model

def create_cnn_model(input_shape, num_classes):
    # Initialize a Sequential model
    model = tf.keras.Sequential()

    # First Convolutional Layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))  # 32 filters, 3x3 kernel
    model.add(tf.keras.layers.BatchNormalization())  # Add batch normalization for faster convergence
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer with 2x2 pool size
    model.add(tf.keras.layers.Dropout(0.25))  # Add dropout layer to reduce overfitting

    # Second Convolutional Layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))  # 64 filters, 3x3 kernel
    model.add(tf.keras.layers.BatchNormalization())  # Add batch normalization
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Another max pooling layer
    model.add(tf.keras.layers.Dropout(0.25))  # Add dropout layer to reduce overfitting

    # Flattening Layer
    model.add(tf.keras.layers.Flatten())  # Flatten the 2D feature maps to a 1D vector

    # Fully Connected Layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # Fully connected layer with 128 neurons
    model.add(tf.keras.layers.Dropout(0.5))  # Add dropout layer before the output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # Output layer with softmax activation for classification

    return model  # Return the constructed model


# If this script is run directly, we can test the model creation
if __name__ == '__main__':
    # Get the input shape from the CIFAR-10 dataset as seen in explore_dataset.py
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    num_classes = 10  # Number of classes in CIFAR-10 dataset

    # Create the CNN model
    cnn_model = create_cnn_model(input_shape, num_classes)

    # Print the model summary to verify the architecture
    cnn_model.summary()  # Display the model architecture summary

    # Example of loading and preprocessing images for training
    # x_train, y_train = ... # Load your training dataset here
    # x_train_preprocessed = normalize_images(x_train)  # Normalize the training images
    # augmentor = augment_images(x_train_preprocessed)  # Prepare the augmentor for the training images
