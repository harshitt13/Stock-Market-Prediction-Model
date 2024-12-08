import pandas as pd
import os

def combine_predictions():
    # Load the two CSV files containing predictions
    lr_predictions_path = 'data/future_predictions_lr.csv'
    lstm_predictions_path = 'data/future_predictions_lstm.csv'

    # Check if both files exist
    if os.path.exists(lr_predictions_path) and os.path.exists(lstm_predictions_path):
        # Read the CSV files
        lr_predictions = pd.read_csv(lr_predictions_path)
        lstm_predictions = pd.read_csv(lstm_predictions_path)

        # Print columns to help debug
        print("Linear Regression Columns:", lr_predictions.columns)
        print("LSTM Model Columns:", lstm_predictions.columns)

        # Check if 'date' column exists in both dataframes
        if 'date' in lr_predictions.columns and 'date' in lstm_predictions.columns:
            # Assuming the date column is 'date' (replace 'date' with the correct column name)
            combined_predictions = pd.merge(lr_predictions, lstm_predictions, how='outer', on='date')
            
            # Ensure the data directory exists
            os.makedirs('data', exist_ok=True)

            # Save the combined data to a new CSV file
            combined_predictions_path = 'data/combined_predictions.csv'
            combined_predictions.to_csv(combined_predictions_path, index=False)

            print(f"Combined predictions saved to '{combined_predictions_path}'")
        else:
            print("The 'date' column is missing in one or both prediction files.")
    else:
        print("One or both prediction files do not exist.")

# Run the function to combine predictions
combine_predictions()
