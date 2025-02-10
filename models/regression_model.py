import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, Tuple, Union

class CS2RegressionModel:
    def __init__(self, model_path: str = 'models/regression_model.pkl'):
        """
        Initialize the CS2 regression model.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'kills', 'headshots', 'awp_kills', 'first_bloods',
            'headshot_percentage', 'kills_bias', 'headshots_bias',
            'awp_kills_bias', 'first_bloods_bias'
        ]
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        # Ensure all required features are present
        missing_features = [col for col in self.feature_columns 
                          if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        return data[self.feature_columns]
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, 
             random_state: int = 42) -> Tuple[float, float]:
        """
        Train the regression model.
        
        Args:
            data: Training data DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (MSE, R² score)
        """
        # Prepare features and target
        X = self.prepare_features(data)
        y = data['prizepicks_score']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model and scaler
        self.save_model()
        
        return mse, r2
    
    def predict_score(self, player_stats: Dict[str, float]) -> float:
        """
        Predict PrizePicks score for a player.
        
        Args:
            player_stats: Dictionary containing player statistics
            
        Returns:
            Predicted PrizePicks score
        """
        if self.model is None:
            self.load_model()
        
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([player_stats])
        
        # Prepare features
        X = self.prepare_features(input_df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Ensure prediction is non-negative
        return max(0.0, prediction)
    
    def save_model(self) -> None:
        """Save the trained model and scaler to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, self.model_path)
        
        print(f"Model saved to: {self.model_path}")
    
    def load_model(self) -> None:
        """Load the trained model and scaler from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model and scaler
        saved_data = joblib.load(self.model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_columns = saved_data['feature_columns']
        
        print(f"Model loaded from: {self.model_path}")

if __name__ == '__main__':
    # Example usage
    try:
        # Load processed data
        data = pd.read_csv('data/processed/processed_stats.csv')
        
        # Initialize and train model
        model = CS2RegressionModel()
        mse, r2 = model.train(data)
        print(f"Model trained successfully!")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Example prediction
        example_stats = {
            'kills': 20,
            'headshots': 10,
            'awp_kills': 5,
            'first_bloods': 2,
            'headshot_percentage': 50.0,
            'kills_bias': 1.5,
            'headshots_bias': 0.8,
            'awp_kills_bias': 0.3,
            'first_bloods_bias': 0.1
        }
        
        predicted_score = model.predict_score(example_stats)
        print(f"Predicted PrizePicks Score: {predicted_score:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
