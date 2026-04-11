import unittest
import os
import pandas as pd
import torch
from lstm_model import train_lstm_model

class TestTrainLSTMModel(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='2023-01-01', periods=120)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Close': [150 + i for i in range(120)],
            'High': [155 + i for i in range(120)],
            'Low': [145 + i for i in range(120)],
            'Open': [148 + i for i in range(120)],
            'Volume': [1000000 + i * 1000 for i in range(120)],
        })

        self.model_dir = 'models'
        self.data_dir = 'data'
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'lstm_model.pt')
        self.predictions_path = os.path.join(self.data_dir, 'future_predictions_lstm.csv')

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.predictions_path):
            os.remove(self.predictions_path)

    def test_train_lstm_model(self):
        result = train_lstm_model(self.test_data, future_days=5)

        self.assertTrue(os.path.exists(self.model_path), "The LSTM model file should be saved.")
        self.assertTrue(os.path.exists(self.predictions_path), "The future predictions file should be saved.")

        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIn('future_predictions', result)
        
        self.assertEqual(len(result['future_predictions']), 5)

if __name__ == '__main__':
    unittest.main()
