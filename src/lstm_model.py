import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the LSTM Model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Save the model and scaler
def save_lstm_model(model, scaler, model_path, scaler_path):
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {model_path} and scaler saved to {scaler_path}.")


# Load the model and scaler
def load_lstm_model(model_path, scaler_path, input_size, hidden_size, num_layers, output_size):
    # Recreate the model architecture
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# Forecast future prices
def forecast_future(model, scaler, last_known_data, forecast_days):
    model.eval()  # Ensure the model is in evaluation mode

    predictions = []
    input_data = last_known_data

    for _ in range(forecast_days):
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

        # Generate the prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        predictions.append(prediction)

        # Append the prediction to input data for the next step
        next_input = np.append(input_data[0][1:], prediction).reshape(1, -1)
        input_data = next_input

    return predictions
