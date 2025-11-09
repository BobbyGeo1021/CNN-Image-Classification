import unittest  # Import the unittest framework
import json  # Import json for handling JSON data
from deploy_model import app  # Import the Flask app from deploy_model.py

class TestDeployment(unittest.TestCase):  # Define a test case class for testing the deployment

    def setUp(self):  # Method to set up test variables
        self.app = app.test_client()  # Create a test client for the Flask app
        self.app.testing = True  # Set testing mode for the app

    def test_predict_valid_image(self):  # Test case for valid image prediction
        with open('path/to/test_image.jpg', 'rb') as img:  # Open a test image file
            response = self.app.post('/predict', data={'file': img})  # Send a POST request to the /predict endpoint
        self.assertEqual(response.status_code, 200)  # Assert that the response status code is 200
        data = json.loads(response.data)  # Load the response data
        self.assertIn('predicted_class', data)  # Assert that the predicted_class is in the response

    def test_predict_no_file(self):  # Test case for no file provided
        response = self.app.post('/predict')  # Send a POST request without a file
        self.assertEqual(response.status_code, 400)  # Assert that the response status code is 400
        data = json.loads(response.data)  # Load the response data
        self.assertIn('error', data)  # Assert that an error message is included in the response

    def test_predict_invalid_file(self):  # Test case for invalid file
        response = self.app.post('/predict', data={'file': (None, 'invalid.txt')})  # Send a POST request with an invalid file
        self.assertEqual(response.status_code, 400)  # Assert that the response status code is 400
        data = json.loads(response.data)  # Load the response data
        self.assertIn('error', data)  # Assert that an error message is included in the response

if __name__ == '__main__':  # Check if this script is run directly
    unittest.main()  # Run all the test cases
