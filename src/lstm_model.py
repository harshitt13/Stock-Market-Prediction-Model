import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
# import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(data):
    """
    Train and evaluate an LSTM model on the given stock data.

    Args:
        data (pd.DataFrame): The stock data to train the model on.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Select features and target variable
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # Prepare the data for LSTM
    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), :]
            X.append(a)
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], len(features) - 1))), axis=1))[:, 0]
    y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features) - 1))), axis=1))[:, 0]

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Close Prices')
    plt.ylabel('Predicted Close Prices')
    plt.title('Actual vs Predicted Close Prices using LSTM')
    plt.savefig('images/lstm_actual_vs_predicted.png')
    plt.show()

    # Save the trained model
    model_path = 'models/lstm_model.keras'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Predict future stock prices
    future_dates = pd.date_range(start=data.index.max(), periods=30, freq='B')  # Predict for the next 30 business days
    future_data = pd.DataFrame(index=future_dates, columns=features)

    # Assuming the future data is not available, we will use the last available data for prediction
    last_available_data = data[features].iloc[-1]

    for feature in features:
        future_data[feature] = last_available_data[feature]

    scaled_future_data = scaler.transform(future_data)
    X_future = []

    for i in range(len(scaled_future_data) - time_step):
        X_future.append(scaled_future_data[i:(i + time_step), :])

    X_future = np.array(X_future)
    if len(X_future) > 0:
        future_predictions = model.predict(X_future)
    else:
        future_predictions = np.array([])

    if future_predictions.size > 0:
        future_predictions = scaler.inverse_transform(np.concatenate((future_predictions, np.zeros((future_predictions.shape[0], len(features) - 1))), axis=1))[:, 0]
    else:
        future_predictions = np.zeros(len(future_data))

    # Save future predictions to a CSV file
    future_data['Predicted Close'] = future_predictions
    future_data.to_csv('data/future_predictions_lstm.csv')

    print("Future stock price predictions saved to 'future_predictions_lstm.csv'")
