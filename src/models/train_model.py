"""Model Training and Evaluation Module."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate multiple regression models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize ModelTrainer."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.logger = logger
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all regression models."""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'Support Vector Regression': SVR(kernel='rbf', C=100, epsilon=0.1),
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state, verbosity=0)
        
        return models
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str) -> Any:
        """Train a single model."""
        models = self._initialize_models()
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = models[model_name]
        model.fit(X_train, y_train)
        self.models[model_name] = model
        self.logger.info(f"Trained {model_name}")
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all available models."""
        models = self._initialize_models()
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            self.logger.info(f"Trained {name}")
        
        return self.models
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2 Score': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred)
        }
        
        if model_name:
            self.results[model_name] = metrics
            self.logger.info(f"Evaluated {model_name}")
        
        return metrics
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all trained models."""
        if not self.models:
            raise ValueError("No models trained yet.")
        
        for name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, name)
        
        results_df = pd.DataFrame(self.results).T
        self.logger.info(f"Evaluated {len(self.models)} models")
        
        return results_df.sort_values('R2 Score', ascending=False)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, model_name: str,
                      cv: int = 5) -> Tuple[float, float]:
        """Perform k-fold cross-validation."""
        models = self._initialize_models()
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        self.logger.info(f"Cross-validation for {model_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean(), scores.std()
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """Save trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        joblib.dump(self.models[model_name], filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str) -> Any:
        """Load trained model from disk."""
        model = joblib.load(filepath)
        self.models[model_name] = model
        self.logger.info(f"Model loaded from {filepath}")
        return model
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name].predict(X)
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all model evaluations."""
        if not self.results:
            raise ValueError("No evaluation results available")
        
        return pd.DataFrame(self.results).T.sort_values('R2 Score', ascending=False)
