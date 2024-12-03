import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import LSTM model class
from lstm_model import LSTMStockModel

# Import helper functions from fetch_data.py
from fetch_data import preprocess_data

# Define constants
DATA_PATH = "data/stock_data.csv"
LINEAR_MODEL_PATH = "models/linear_regression_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.p5"

def load_models():
    """
    Load the trained models: Linear Regression and LSTM.
    """
    # Load Linear Regression model
    with open(LINEAR_MODEL_PATH, 'rb') as file:
        linear_model = pickle.load(file)

    # Load LSTM model
    lstm_model = LSTMStockModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))

    return linear_model, lstm_model

def evaluate_model(model, X_test, y_test, model_type="Linear Regression"):
    """
    Evaluate a model on the test data.
    """
    if model_type == "Linear Regression":
        y_pred = model.predict(X_test)
    elif model_type == "LSTM":
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
            y_pred = model(X_test_tensor).squeeze().numpy()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Performance of {model_type}:")
    print(f"  Mean Squared Error (MSE): {mse}")
    print(f"  Mean Absolute Error (MAE): {mae}\n")

    return y_pred

def plot_predictions(dates, y_test, y_pred_linear, y_pred_lstm):
    """
    Plot actual vs predicted prices for both models.
    """
    plt.figure(figsize=(12, 6))

    # Actual prices
    plt.plot(dates, y_test, label="Actual Prices", color='blue', linewidth=2)

    # Linear Regression predictions
    plt.plot(dates, y_pred_linear, label="Linear Regression Predictions", color='green', linestyle='--')

    # LSTM predictions
    plt.plot(dates, y_pred_lstm, label="LSTM Predictions", color='orange', linestyle='--')

    plt.title("Stock Price Predictions: Actual vs Predicted")
    plt.xlabel("Dates")
    plt.ylabel("Stock Prices")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    Main script to compare and evaluate Linear Regression and LSTM models.
    """
    print("Loading data...")
    # Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, dates = preprocess_data(df)

    print("Loading trained models...")
    linear_model, lstm_model = load_models()
    _, X_test, _, y_test, dates = preprocess_data(df)
    print("Evaluating Linear Regression model...")
    y_pred_linear = evaluate_model(linear_model, X_test, y_test, model_type="Linear Regression")

    print("Evaluating LSTM model...")
    y_pred_lstm = evaluate_model(lstm_model, X_test, y_test, model_type="LSTM")

    print("Plotting predictions...")
    plot_predictions(dates, y_test, y_pred_linear, y_pred_lstm)

if __name__ == "__main__":
    main()
