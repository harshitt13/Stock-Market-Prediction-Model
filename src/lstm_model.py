import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


class LSTMModel(nn.Module):
    """
    LSTM Model for stock price prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the output through the fully connected layer
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out


def save_lstm_model(model, scaler, model_path, scaler_path):
    """
    Save the trained LSTM model and the associated scaler.

    Args:
        model (LSTMModel): Trained LSTM model.
        scaler (StandardScaler): Scaler used for feature scaling.
        model_path (str): Path to save the model.
        scaler_path (str): Path to save the scaler.
    """
    try:
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Model saved to {model_path} and scaler saved to {scaler_path}.")
    except Exception as e:
        print(f"Error saving model or scaler: {e}")


def load_lstm_model(model_path, scaler_path, input_size, hidden_size, num_layers, output_size):
    """
    Load the trained LSTM model and the scaler.

    Args:
        model_path (str): Path to the saved model.
        scaler_path (str): Path to the saved scaler.
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        output_size (int): Number of output features.

    Returns:
        tuple: Loaded LSTM model and scaler.
    """
    try:
        # Initialize the model with the same architecture
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print(f"Model and scaler loaded successfully from {model_path} and {scaler_path}.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None


def forecast_future(model, scaler, last_known_data, forecast_days):
    """
    Forecast future stock prices using the trained LSTM model.

    Args:
        model (LSTMModel): Trained LSTM model.
        scaler (StandardScaler): Scaler used during training.
        last_known_data (np.ndarray): Last known stock data as input (2D array).
        forecast_days (int): Number of future days to forecast.

    Returns:
        list: Predicted stock prices for the forecast period.
    """
    try:
        model.eval()  # Ensure the model is in evaluation mode
        predictions = []
        input_data = last_known_data

        for _ in range(forecast_days):
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

            # Move to the same device as the model
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)

            # Generate the prediction
            with torch.no_grad():
                prediction = model(input_tensor).item()

            predictions.append(prediction)

            # Prepare the next input by appending the prediction
            next_input = np.append(input_data[0][1:], prediction).reshape(1, -1)
            input_data = next_input

        return predictions
    except Exception as e:
        print(f"Error during forecasting: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Parameters
    input_size = 5  # Example: Number of features
    hidden_size = 50
    num_layers = 2
    output_size = 1
    model_path = "models/lstm_model.pth"
    scaler_path = "models/lstm_scaler.pkl"

    # Load a trained model and scaler
    model, scaler = load_lstm_model(model_path, scaler_path, input_size, hidden_size, num_layers, output_size)

    # Example: Forecasting future prices
    if model and scaler:
        # Example last known data (replace with actual data)
        last_known_data = np.array([[100, 101, 99, 100, 5000]])  # Adjust based on your features
        forecast_days = 5  # Predict the next 5 days

        predictions = forecast_future(model, scaler, last_known_data, forecast_days)
        print("Predicted Prices:", predictions)
