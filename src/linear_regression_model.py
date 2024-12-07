import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the Linear Regression model for training and prediction
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def train(self, X, y):
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        # Fit the linear regression model
        self.model.fit(X_scaled, y)

    def predict(self, X):
        # Standardize features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, filepath):
        # Save the model and scaler to a file using pickle
        with open(filepath, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)

    @classmethod
    def load(cls, filepath):
        # Load the model and scaler from a file using pickle
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            model = cls()
            model.model = data['model']
            model.scaler = data['scaler']
            return model

# Function to prepare data for training the linear regression model
def prepare_data(data):
    # Handle missing values and preprocess data
    data.fillna(method='ffill', inplace=True)

    # Use the 'Close' price as the target variable
    X = data.drop(columns=['Close', 'Date'])
    y = data['Close']
    
    return X, y

def train_linear_regression(file_path, save_path):
    # Load and preprocess data
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)

    # Prepare features and target for training
    X, y = prepare_data(data)
    
    # Initialize and train the model
    model = LinearRegressionModel()
    model.train(X, y)
    
    # Save the trained model
    model.save(save_path)
    print(f"Linear Regression model saved to {save_path}")

    return model

def predict_linear_regression(model, X):
    # Predict using the trained Linear Regression model
    return model.predict(X)
