import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LinearRegressionModel:
    """
    A wrapper class for Linear Regression using sklearn with scaling and save/load functionality.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def train(self, X, y):
        """
        Train the Linear Regression model.

        Args:
            X (pd.DataFrame): Features for training.
            y (pd.Series): Target variable (stock prices).
        """
        try:
            X_scaled = self.scaler.fit_transform(X)  # Scale features
            self.model.fit(X_scaled, y)
            print("Model trained successfully.")
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, X):
        """
        Predict using the trained Linear Regression model.

        Args:
            X (pd.DataFrame): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        try:
            X_scaled = self.scaler.transform(X)  # Scale features using fitted scaler
            return self.model.predict(X_scaled)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def save(self, filepath):
        """
        Save the model and scaler to a file.

        Args:
            filepath (str): File path to save the model.
        """
        try:
            with open(filepath, 'wb') as file:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, file)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving the model: {e}")

    @classmethod
    def load(cls, filepath):
        """
        Load the model and scaler from a file.

        Args:
            filepath (str): File path to load the model.

        Returns:
            LinearRegressionModel: Loaded model.
        """
        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                model = cls()
                model.model = data['model']
                model.scaler = data['scaler']
                print(f"Model loaded from {filepath}")
                return model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None


def prepare_data(data):
    """
    Prepare the data for training and prediction.

    Args:
        data (pd.DataFrame): Input stock data.

    Returns:
        tuple: Features (X) and target (y) for the model.
    """
    try:
        # Handle missing values by forward filling
        data.fillna(method='ffill', inplace=True)

        # Ensure required columns are present
        if 'Close' not in data.columns:
            raise ValueError("The 'Close' column is missing from the data.")

        # Separate features and target
        X = data.drop(columns=['Close', 'Date'], errors='ignore')
        y = data['Close']

        return X, y
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None


def train_linear_regression(file_path, save_path):
    """
    Train a Linear Regression model on stock data and save it.

    Args:
        file_path (str): Path to the CSV file containing stock data.
        save_path (str): Path to save the trained model.

    Returns:
        LinearRegressionModel: Trained model.
    """
    try:
        # Load and preprocess data
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Prepare features and target for training
        X, y = prepare_data(data)

        if X is None or y is None:
            raise ValueError("Failed to prepare data for training.")

        # Initialize and train the model
        model = LinearRegressionModel()
        model.train(X, y)

        # Save the trained model
        model.save(save_path)
        return model
    except Exception as e:
        print(f"Error during training process: {e}")
        return None


def predict_linear_regression(model, X):
    """
    Predict stock prices using a trained Linear Regression model.

    Args:
        model (LinearRegressionModel): Trained model.
        X (pd.DataFrame): Features for prediction.

    Returns:
        np.ndarray: Predicted stock prices.
    """
    try:
        return model.predict(X)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Example usage
if __name__ == "__main__":
    file_path = "data/AAPL_stock_data.csv"  # Replace with actual path
    save_path = "models/linear_regression_model.pkl"

    # Train the model
    trained_model = train_linear_regression(file_path, save_path)

    # Example prediction
    if trained_model:
        data = pd.read_csv(file_path)
        X, _ = prepare_data(data)
        predictions = predict_linear_regression(trained_model, X)
        print("Sample Predictions:", predictions[:5])
