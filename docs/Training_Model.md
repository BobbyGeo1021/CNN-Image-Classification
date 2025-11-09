# Training Model

## Overview
This document provides details about the training process for the CNN model used in this project.

## Training Process
1. **Data Generators**: Training and validation data are loaded using data generators, which provide images in batches to minimize memory usage.
2. **Model Compilation**: The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
3. **Callbacks**: The training process includes:
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau to adjust the learning rate dynamically
4. **Training Execution**: The model is trained for a specified number of epochs, with validation metrics monitored to evaluate performance.

## Conclusion
The training process is designed to optimize the model's accuracy while preventing overfitting through careful monitoring and adjustments.