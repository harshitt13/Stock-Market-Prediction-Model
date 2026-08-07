import unittest
import os
import pandas as pd
import numpy as np
from fetch_data import fetch_stock_data, save_to_csv

class TestFetchStockData(unittest.TestCase):
    def setUp(self):
        self.ticker_symbol = 'AAPL'
        self.start_date = '2023-01-01'
        self.end_date = '2024-01-01'
        self.data_dir = 'data'
        self.filepath = os.path.join(self.data_dir, "stock_data.csv")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_fetch_stock_data_valid(self):
        result_df = fetch_stock_data(self.ticker_symbol, self.start_date, self.end_date)
        self.assertIsInstance(result_df, pd.DataFrame, "Result should be a DataFrame.")
        self.assertTrue(os.path.exists(self.filepath), "CSV file should be saved.")
        self.assertFalse(result_df.empty, "DataFrame should not be empty.")
        
        # Verify core columns exist
        expected_columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'MACD', 'RSI_14', 'BB_Upper']
        actual_columns = result_df.columns.tolist()
        for col in expected_columns:
            self.assertIn(col, actual_columns, f"Column '{col}' should exist in the DataFrame.")

    def test_save_to_csv(self):
        data = {
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [150.0, 152.5],
            'Volume': [1000000, 1200000],
        }
        sample_df = pd.DataFrame(data)
        save_to_csv(sample_df, 'TEST')

        self.assertTrue(os.path.exists(self.filepath), "CSV file should be saved.")
        saved_df = pd.read_csv(self.filepath)
        pd.testing.assert_frame_equal(sample_df, saved_df, check_dtype=False)

    def test_fetch_stock_data_invalid_ticker(self):
        invalid_ticker = 'INVALID_TICKER_XYZ'
        result_df = fetch_stock_data(invalid_ticker, self.start_date, self.end_date)
        self.assertIsNone(result_df, "Result should be None for an invalid ticker.")

if __name__ == '__main__':
    unittest.main()
