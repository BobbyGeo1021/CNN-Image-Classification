# Data Preprocessing

## Overview
This document outlines the data preprocessing steps taken to prepare the CIFAR-10 dataset for training the CNN model.

## Steps Involved
1. **Loading Images**: Images are loaded from specified file paths and resized to 32x32 pixels.
2. **Normalization**: Pixel values are normalized to the range [0, 1] to improve model convergence during training.
3. **Data Augmentation**: Various augmentation techniques are applied to the training images to increase dataset diversity and reduce overfitting:
   - Random rotations
   - Width and height shifts
   - Shearing
   - Zooming
   - Horizontal flipping

## Conclusion
These preprocessing steps are crucial for improving the model's performance and generalization capabilities.