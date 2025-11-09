# CNN Architecture

## Overview
This document provides a detailed explanation of the Convolutional Neural Network (CNN) architecture used in this project for image classification.

## Architecture Details
1. **Input Layer**: The input layer accepts images of size 32x32 pixels with 3 color channels (RGB).
2. **Convolutional Layers**: The model includes two convolutional layers:
   - The first layer uses 32 filters with a kernel size of 3x3 and ReLU activation.
   - The second layer uses 64 filters with a kernel size of 3x3 and ReLU activation.
3. **Pooling Layers**: After each convolutional layer, a max pooling layer with a pool size of 2x2 is applied to downsample the feature maps.
4. **Dropout Layers**: Dropout layers are added after the pooling layers to reduce overfitting by randomly setting a fraction of input units to 0 during training.
5. **Flatten Layer**: The feature maps are flattened into a 1D array before passing to the fully connected layers.
6. **Fully Connected Layer**: A dense layer with 128 neurons and ReLU activation is used, followed by another dropout layer.
7. **Output Layer**: The final output layer has 10 neurons (one for each class) with softmax activation to provide class probabilities.

## Summary
This architecture is designed to effectively capture spatial hierarchies in images, making it suitable for the CIFAR-10 classification task.