import numpy as np  # Import NumPy for numerical operations
import tensorflow as tf  # Import TensorFlow library
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation


def load_and_preprocess_image(image_path):
    """Load an image from a file and preprocess it for model input."""
    # Validate the image path input to prevent directory traversal attacks
    if not isinstance(image_path, str) or not image_path:
        raise ValueError('Invalid image path provided.')  # Raise an error if the input is not valid

    # Load the image from the specified path and resize it to 32x32 pixels
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))  
    # Convert the image to a NumPy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)  
    # Normalize the pixel values to the range [0, 1]
    image_array = image_array / 255.0  
    return image_array  # Return the preprocessed image


def normalize_images(images):
    """Normalize a batch of images to the range [0, 1]."""
    # Validate input to ensure images is a NumPy array
    if not isinstance(images, np.ndarray):
        raise ValueError('Input must be a NumPy array.')  # Raise an error if the input is not valid
    # Normalize the images by dividing by 255.0 (the maximum pixel value)
    return images / 255.0  # Return the normalized images


def augment_images(images):
    """Apply data augmentation techniques to a batch of images."""
    # Validate input to ensure images is a NumPy array
    if not isinstance(images, np.ndarray):
        raise ValueError('Input must be a NumPy array.')  # Raise an error if the input is not valid
    # Create an instance of ImageDataGenerator with various augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,  # Randomly rotate images in the range (0-20 degrees)
        width_shift_range=0.2,  # Randomly shift images horizontally (20% of total width)
        height_shift_range=0.2,  # Randomly shift images vertically (20% of total height)
        shear_range=0.2,  # Shear angle in counter-clockwise direction in degrees
        zoom_range=0.2,  # Randomly zoom into images
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill in new pixels that are created during the transformations
    )
    # Fit the generator to the images
    datagen.fit(images)  
    return datagen  # Return the data generator


def create_data_generator(directory, batch_size=32, image_size=(32, 32), shuffle=True):
    """Creates a data generator for loading images in batches from a directory."""
    # Create an instance of ImageDataGenerator for normalization and augmentation
    datagen = ImageDataGenerator(rescale=1./255)  # Rescale images to [0, 1]
    # Use flow_from_directory to create a generator that loads images from the specified directory
    generator = datagen.flow_from_directory(
        directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',  # Use sparse for integer labels
        shuffle=shuffle  # Shuffle data if required
    )
    return generator  # Return the data generator


# Example usage:
if __name__ == '__main__':
    # Load and preprocess a sample image
    sample_image_path = 'path/to/sample_image.jpg'  # Path to a sample image
    processed_image = load_and_preprocess_image(sample_image_path)  # Load and preprocess the sample image
    print(processed_image.shape)  # Print the shape of the processed image

    # Normalize a batch of images (assuming images is a NumPy array of shape (num_samples, 32, 32, 3))
    normalized_images = normalize_images(np.random.rand(10, 32, 32, 3))  # Example batch of random images
    print(normalized_images.shape)  # Print the shape of the normalized images

    # Create an augmentation generator for the images
    augmentor = augment_images(normalized_images)  # Create the augmentation generator
    # The augmentor can now be used in model training to generate augmented images on the fly

    # Example of creating a data generator from a directory
    # generator = create_data_generator('path/to/data', batch_size=32)  # Create a data generator for training images
