import unittest
import os
import pandas as pd
import joblib
from tree_model import train_tree_model

class TestTrainTreeModel(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame mimic for stock data tests
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Close': [150 + i for i in range(100)],
            'High': [155 + i for i in range(100)],
            'Low': [145 + i for i in range(100)],
            'Open': [148 + i for i in range(100)],
            'Volume': [1000000 + i * 1000 for i in range(100)],
            'SMA_20': [149 + i for i in range(100)],
            'SMA_50': [145 + i for i in range(100)],
            'RSI_14': [50 + (i%20) for i in range(100)],
            'Log_Return': [0.01 for _ in range(100)],
            'DayOfWeek': dates.dayofweek,
        })

        self.model_dir = 'models'
        self.data_dir = 'data'
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'tree_ensemble_model.pkl')
        self.predictions_path = os.path.join(self.data_dir, 'future_predictions_tree.csv')

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.predictions_path):
            os.remove(self.predictions_path)

    def test_train_tree_model(self):
        result = train_tree_model(self.test_data, future_days=5)

        self.assertTrue(os.path.exists(self.model_path), "The model file should be saved.")
        self.assertTrue(os.path.exists(self.predictions_path), "The future predictions file should be saved.")
        
        self.assertIn('model', result)
        self.assertIn('scaler', result)
        self.assertIn('y_test', result)
        self.assertIn('y_pred', result)
        self.assertIn('future_predictions', result)
        
        self.assertEqual(len(result['future_predictions']), 5)

if __name__ == '__main__':
    unittest.main()
