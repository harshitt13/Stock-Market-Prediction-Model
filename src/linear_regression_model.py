import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt

def train_linear_regression_model(data):
    """
    Train and evaluate a linear regression model on the given stock data.

    Args:
        data (pd.DataFrame): The stock data to train the model on.
    """
    # Select features and target variable
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']
    X = data[features]
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Close Prices')
    plt.ylabel('Predicted Close Prices')
    plt.title('Actual vs Predicted Close Prices using Linear Regression')
    plt.savefig('images/lr_actual_vs_predicted.png')
    plt.show()

    # Save the trained model
    model_path = 'models/linear_regression_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Predict future stock prices
    future_dates = pd.date_range(start=data['Date'].max(), periods=30, freq='B')  # Predict for the next 30 business days
    future_data = pd.DataFrame(index=future_dates, columns=features)

    # Assuming the future data is not available, we will use the last available data for prediction
    last_available_data = data[features].iloc[-1]

    for feature in features:
        future_data[feature] = last_available_data[feature]

    future_predictions = model.predict(future_data)

    # Save future predictions to a CSV file
    future_data['Predicted Close'] = future_predictions
    future_data.to_csv('data/future_predictions_lr.csv')

    print("Future stock price predictions saved to 'future_predictions_using_lr.csv'")
