import os  # Import the os module for handling file paths
import numpy as np  # Import NumPy for numerical operations
import tensorflow as tf  # Import TensorFlow library
from tensorflow.keras.models import load_model  # Import load_model for loading the trained model
from flask import Flask, request, jsonify  # Import Flask for creating the web server and handling requests

# Initialize a Flask web application
app = Flask(__name__)  # Create an instance of the Flask class for the web server

# Load the trained CNN model from the saved file
model = load_model('path/to/saved_model.h5')  # Load your trained model from the specified path

@app.route('/predict', methods=['POST'])  # Define the /predict endpoint and specify that it accepts POST requests
def predict():  # Define the prediction function
    # Check if an image file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400  # Return error if no file is provided
    file = request.files['file']  # Get the file from the request
    # Ensure the file is an image
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # Return error if no file is selected
    # Load and preprocess the image for prediction
    image = load_and_preprocess_image(file)  # Use the function to load and preprocess the image
    # Expand dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)  # Add batch dimension to the image
    # Make predictions using the model
    predictions = model.predict(image)  # Get predictions from the model
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    return jsonify({'predicted_class': int(predicted_class[0])})  # Return the predicted class as a JSON response

if __name__ == '__main__':  # Check if this script is run directly
    app.run(host='0.0.0.0', port=5000)  # Start the Flask web server on port 5000
