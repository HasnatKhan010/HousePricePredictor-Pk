"""House Price Predictor - ML Project for Real Estate Valuation.

A comprehensive machine learning project for predicting residential property prices
in Pakistan using advanced regression algorithms and feature engineering techniques.

Author: Hasnat Khan
Year: 2026

"""

__version__ = "1.0.0"
__author__ = "Hasnat Khan"
__description__ = "House Price Prediction using Machine Learning"

from src.data.data_loader import DataLoader
from src.features.build_features import FeatureEngineer
from src.models.train_model import ModelTrainer

__all__ = ['DataLoader', 'FeatureEngineer', 'ModelTrainer']
