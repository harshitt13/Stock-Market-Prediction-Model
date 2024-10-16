# Description: This script trains a linear regression model to predict stock prices based on historical data.

# Libraries Used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os


def main():
    # Load the data from the CSV File
    data = pd.read_csv('data/stock_data.csv')

    # Display the first few rows to check the data
    print(data.head())

    # Preprocessing the data
    # Drop any rows with missing values
    data = data.dropna()

    # Define the features (independent variables) and the target (dependent variable)
    features = ['Open', 'High', 'Low', 'Volume', 'Revenue', 'EPS', 'ROE', 'P/E']
    X = data[features]
    y = data['Close']

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict the stock prices on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Plot the results to see how well the model fits the data
    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(len(y_test)), y_test, color='blue', label='Actual Prices')
    plt.scatter(np.arange(len(y_pred)), y_pred, color='red', label='Predicted Prices')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Closing Price')
    plt.legend()
    plt.show()

    # Save the model predictions
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('data/linear_regression_predictions.csv', index=False)
    print('Predictions saved to data/linear_regression_predictions.csv')

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the trained model
    joblib.dump(model, 'models/linear_regression_model.pkl')
    print('Model saved to models/linear_regression_model.pkl')

if __name__ == "__main__":
    main()
