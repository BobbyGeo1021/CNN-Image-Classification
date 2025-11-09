import unittest  # Import the unittest framework
from cnn_model import create_cnn_model  # Import the CNN model creation function
from data_preprocessing import normalize_images  # Import the normalization function
from train_model import train_model  # Import the training function


class TestCNNModel(unittest.TestCase):  # Define a test case class that inherits from unittest.TestCase

    def setUp(self):  # Method to set up test variables
        self.input_shape = (32, 32, 3)  # Define the input shape for the model
        self.num_classes = 10  # Define the number of classes
        self.model = create_cnn_model(self.input_shape, self.num_classes)  # Create the CNN model

    def test_model_output_shape(self):  # Test case to check the output shape of the model
        # Build the model to get output shape
        output_shape = self.model.output_shape  # Get the output shape of the model
        expected_shape = (None, self.num_classes)  # Expected output shape
        self.assertEqual(output_shape, expected_shape)  # Assert that the output shape is as expected

    def test_model_compilation(self):  # Test case to check if the model compiles correctly
        try:
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile the model
            compiled = True  # Set flag to true if compilation does not raise an error
        except Exception as e:
            compiled = False  # Set flag to false if an error occurs
        self.assertTrue(compiled)  # Assert that the model compiled successfully

    def test_layer_configuration(self):  # Test case to check specific layer configurations
        # Check the first convolutional layer
        first_conv_layer = self.model.layers[0]  # Access the first layer
        self.assertEqual(first_conv_layer.filters, 32)  # Assert that the number of filters is 32
        self.assertEqual(first_conv_layer.kernel_size, (3, 3))  # Assert that the kernel size is 3x3
        self.assertEqual(first_conv_layer.activation.__name__, 'relu')  # Assert that activation function is relu


class TestDataPreprocessing(unittest.TestCase):  # Define a test case class for data preprocessing tests

    def test_normalize_images(self):  # Test case for normalization function
        images = np.array([[[0, 0, 0], [255, 255, 255]], [[128, 128, 128], [64, 64, 64]]])  # Sample images
        normalized_images = normalize_images(images)  # Normalize the sample images
        expected_output = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.50196078, 0.50196078, 0.50196078], [0.25098039, 0.25098039, 0.25098039]]])  # Expected normalized output
        np.testing.assert_array_almost_equal(normalized_images, expected_output)  # Assert that the normalized images match the expected output

    def test_augment_images(self):  # Test case for augmentation function
        images = np.random.rand(10, 32, 32, 3)  # Create random images
        augmentor = augment_images(images)  # Create the augmentor
        self.assertIsNotNone(augmentor)  # Assert that the augmentor is created


class TestTrainModel(unittest.TestCase):  # Define a test case class for training tests

    def test_train_model(self):  # Test case for the training function
        # Mock data (in a real test, you would use actual data)
        x_train = np.random.rand(100, 32, 32, 3)  # Mock training images
        y_train = np.random.randint(0, 10, 100)  # Mock training labels
        x_val = np.random.rand(20, 32, 32, 3)  # Mock validation images
        y_val = np.random.randint(0, 10, 20)  # Mock validation labels
        try:
            trained_model = train_model(create_cnn_model((32, 32, 3), 10), x_train, y_train, x_val, y_val, epochs=1)  # Train the model
            model_trained = True  # Set flag to true if training does not raise an error
        except Exception as e:
            model_trained = False  # Set flag to false if an error occurs
        self.assertTrue(model_trained)  # Assert that the model trained successfully


if __name__ == '__main__':  # Check if this script is the main program
    unittest.main()  # Run all the test cases
