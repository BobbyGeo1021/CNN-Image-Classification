## Deployment Instructions

To deploy the trained CNN model, follow these steps:

1. **Create a Deployment Script**:
   Ensure you have the `deploy_model.py` script in your project directory.

2. **Run the Deployment**:
   To start the web server, run:
   ```bash
   python deploy_model.py
   ```
   This will start the server on `http://localhost:5000`.

3. **Make Predictions**:
   You can use a tool like Postman or cURL to send a POST request to the `/predict` endpoint with an image file. For example:
   ```bash
   curl -X POST -F 'file=@path/to/your/image.jpg' http://localhost:5000/predict
   ```
   The server will respond with the predicted class.

4. **Testing the Deployment**:
   To run the tests for the deployment, execute:
   ```bash
   python test_deployment.py
   ```
   This will run the unit tests to ensure the deployment is functioning correctly.
