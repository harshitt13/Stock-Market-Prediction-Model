import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

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


# Train the LSTM model
def train_lstm_model(data, model_path, scaler_path, input_size=1, hidden_size=100, num_layers=2, output_size=1):
    # Prepare the data
    close_prices = data["Close"].values
    scaler = StandardScaler()
    close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    # Create the dataset for training (using previous 60 days to predict the next day)
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(close_prices_scaled)
    X = X.reshape(X.shape[0], X.shape[1], input_size)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Initialize the LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.flatten(), y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model and scaler
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_path} and scaler saved to {scaler_path}.")
    return model, scaler


# Load the LSTM model and scaler
def load_lstm_model(model_path, scaler_path, input_size, hidden_size, num_layers, output_size):
    # Recreate the model architecture
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# Forecast future prices using the trained LSTM model
def forecast_future(model, scaler, last_known_data, forecast_days):
    model.eval()  # Ensure the model is in evaluation mode

    predictions = []
    input_data = last_known_data

    for _ in range(forecast_days):
        # Only scale the last feature (closing price), not the entire array
        input_scaled = scaler.transform(input_data[:, -1].reshape(-1, 1))  # Reshape to 2D array for scaling
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

        # Generate the prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        predictions.append(prediction)

        # Append the prediction to input data for the next step
        next_input = np.append(input_data[0][1:], prediction).reshape(1, -1)
        input_data = next_input

    return predictions
