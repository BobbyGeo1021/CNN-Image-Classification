# Step 2: Explore the Dataset

# Import necessary libraries for data handling and visualization
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
# This function returns the training and testing data as tuples of images and labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Print the shapes of the datasets
# x_train contains the training images and y_train contains the corresponding labels
print(f"x_train shape: {x_train.shape}")  # Should be (50000, 32, 32, 3)
print(f"y_train shape: {y_train.shape}")  # Should be (50000, 1)
print(f"x_test shape: {x_test.shape}")    # Should be (10000, 32, 32, 3)
print(f"y_test shape: {y_test.shape}")    # Should be (10000, 1)

# Print the number of classes
# CIFAR-10 has 10 unique classes
num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")  # Should print 10

# Visualizing some sample images
# This function takes images and their labels and displays them in a grid

def show_sample_images(images, labels):
    plt.figure(figsize=(10, 10))  # Set the figure size for the plot
    for i in range(9):  # Display 9 images in a 3x3 grid
        plt.subplot(3, 3, i + 1)  # Create a subplot for each image
        plt.imshow(images[i])  # Display the image
        plt.title(f"Label: {labels[i][0]}")  # Set the title to the label of the image
        plt.axis("off")  # Hide the axes
    plt.show()  # Render the visualization

# Call the function to show sample images from the training set
show_sample_images(x_train, y_train)  # Display sample images and their labels
