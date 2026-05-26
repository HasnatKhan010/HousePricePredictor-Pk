"""
Model Training and Evaluation Module
Handles model development, training, and prediction
"""

import logging
import pickle
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Machine Learning model for price prediction
    
    Attributes:
        models (Dict): Dictionary of trained models
        best_model: Best performing model
        feature_names (list): List of feature names
    """
    
    def __init__(self):
        """Initialize the price predictor"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.scaler = None
        
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    random_state: int = 42) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate performance
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Test set size
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict: Performance metrics for each model
        """
        logger.info("Starting model training...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        
        # Define models
        model_configs = {
            "CatBoost": CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                verbose=False,
                random_state=random_state
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=random_state,
                n_jobs=-1
            ),
            "LinearRegression": LinearRegression()
        }
        
        # Train and evaluate models
        results = {}
        for model_name, model in model_configs.items():
            logger.info(f"Training {model_name}...")
            
            # Train
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                "R2_Score": r2,
                "RMSE": rmse,
                "MAE": mae
            }
            
            logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # Select best model
        self.best_model_name = max(results, key=lambda x: results[x]["R2_Score"])
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model selected: {self.best_model_name}")
        logger.info(f"Results: {results[self.best_model_name]}")
        
        return results
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the best model
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            "R2_Score": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(f"Model Evaluation Metrics: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
            
        return self.best_model.predict(X)
    
    def save_model(self, filepath: str):
        """
        Save the best model to file
        
        Args:
            filepath (str): Path to save model
        """
        if self.best_model is None:
            raise ValueError("No model to save")
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath (str): Path to model file
        """
        with open(filepath, 'rb') as f:
            self.best_model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            
        Returns:
            Dict: Cross-validation scores
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
            
        scores = cross_val_score(self.best_model, X, y, cv=cv, scoring='r2')
        
        cv_metrics = {
            "Mean_R2": scores.mean(),
            "Std_R2": scores.std(),
            "Individual_Scores": scores.tolist()
        }
        
        logger.info(f"Cross-validation Results: Mean R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
        return cv_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model
        
        Returns:
            Dict: Feature importance scores
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
            
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            logger.warning("Model does not support feature importance")
            return {}


if __name__ == "__main__":
    logger.info("Model module ready for use")
