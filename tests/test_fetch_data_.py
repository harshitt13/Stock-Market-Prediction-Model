import unittest
import os
import pandas as pd
from fetch_data import fetch_stock_data, save_to_csv

class TestFetchStockData(unittest.TestCase):
    def setUp(self):
        """
        Setup runs before every test. We use it to define parameters
        and ensure the `data` directory is clean.
        """
        self.ticker_symbol = 'AAPL'
        self.start_date = '2010-01-01'
        self.end_date = None
        self.data_dir = 'data'
        self.filepath = os.path.join(self.data_dir, "stock_data.csv")

        # Create `data` directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def tearDown(self):
        """
        Cleanup runs after every test. We use it to remove any
        test artifacts (e.g., CSV files).
        """
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_fetch_stock_data_valid(self):
        """
        Test the `fetch_stock_data` function with valid inputs.
        """
        result_df = fetch_stock_data(self.ticker_symbol, self.start_date, self.end_date)
        # Check if result is a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame, "Result should be a DataFrame.")
        # Check if CSV file is saved
        self.assertTrue(os.path.exists(self.filepath), "CSV file should be saved.")
        # Check if the DataFrame is not empty
        self.assertFalse(result_df.empty, "DataFrame should not be empty.")
        # Verify specific columns
        expected_columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']
        actual_columns = result_df.columns.tolist()
        for col in expected_columns:
            self.assertIn(col, actual_columns, f"Column '{col}' should exist in the DataFrame.")

    def test_save_to_csv(self):
        """
        Test the `save_to_csv` function to ensure it saves the DataFrame correctly.
        """
        # Create a small sample DataFrame
        data = {
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [150.0, 152.5],
            'Volume': [1000000, 1200000],
            'EPS': [5.0, 5.0],
            'Revenue': [500000000, 500000000],
            'ROE': [0.2, 0.2],
            'P/E': [30.0, 30.0]
        }
        sample_df = pd.DataFrame(data)
        save_to_csv(sample_df, self.ticker_symbol)

        # Check if the file is saved
        self.assertTrue(os.path.exists(self.filepath), "CSV file should be saved.")
        # Read the saved file and compare
        saved_df = pd.read_csv(self.filepath)
        pd.testing.assert_frame_equal(sample_df, saved_df, "Saved DataFrame should match the original.")

    def test_fetch_stock_data_invalid_ticker(self):
        """
        Test the `fetch_stock_data` function with an invalid ticker symbol.
        """
        invalid_ticker = 'INVALID'
        result_df = fetch_stock_data(invalid_ticker, self.start_date, self.end_date)
        # Result should be None for invalid ticker
        self.assertIsNone(result_df, "Result should be None for an invalid ticker.")

if __name__ == '__main__':
    unittest.main()
