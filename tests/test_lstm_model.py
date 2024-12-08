import unittest
import os
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from lstm_model import train_lstm_model

class TestTrainLSTMModel(unittest.TestCase):
    def setUp(self):
        """
        Set up necessary files and test data for testing.
        """
        # Create a sample DataFrame mimicking stock data
        self.test_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=120),
            'Close': [150 + i for i in range(120)],
            'High': [155 + i for i in range(120)],
            'Low': [145 + i for i in range(120)],
            'Open': [148 + i for i in range(120)],
            'Volume': [1000000 + i * 1000 for i in range(120)],
            'EPS': [5.0 + i * 0.01 for i in range(120)],
            'Revenue': [500000000 + i * 1000000 for i in range(120)],
            'ROE': [0.2 + i * 0.001 for i in range(120)],
            'P/E': [30.0 + i * 0.05 for i in range(120)],
        })

        # Create necessary directories
        self.model_dir = 'models'
        self.images_dir = 'images'
        self.data_dir = 'data'

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'lstm_model.keras')
        self.predictions_path = os.path.join(self.data_dir, 'future_predictions_lstm.csv')
        self.plot_path = os.path.join(self.images_dir, 'lstm_actual_vs_predicted.png')

    def tearDown(self):
        """
        Clean up any artifacts created during testing.
        """
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.predictions_path):
            os.remove(self.predictions_path)
        if os.path.exists(self.plot_path):
            os.remove(self.plot_path)

    def test_train_lstm_model(self):
        """
        Test the `train_lstm_model` function with valid data.
        """
        # Run the function
        train_lstm_model(self.test_data)

        # Check if the model was saved
        self.assertTrue(os.path.exists(self.model_path), "The LSTM model file should be saved.")
        # Load the model to verify it was trained
        model = load_model(self.model_path)
        self.assertEqual(model.input_shape[-1], len(['Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']), "The model should have the correct input shape.")

        # Check if the future predictions CSV was saved
        self.assertTrue(os.path.exists(self.predictions_path), "The future predictions file should be saved.")
        future_predictions = pd.read_csv(self.predictions_path)
        self.assertFalse(future_predictions.empty, "The future predictions CSV should not be empty.")

        # Check if the plot was saved
        self.assertTrue(os.path.exists(self.plot_path), "The plot file should be saved.")

    def test_empty_data(self):
        """
        Test the function with an empty DataFrame to ensure it handles this gracefully.
        """
        empty_data = pd.DataFrame(columns=['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E'])
        with self.assertRaises(ValueError):
            train_lstm_model(empty_data)

    def test_missing_columns(self):
        """
        Test the function with a DataFrame missing required columns.
        """
        incomplete_data = self.test_data.drop(columns=['Close'])
        with self.assertRaises(KeyError):
            train_lstm_model(incomplete_data)

if __name__ == '__main__':
    unittest.main()
