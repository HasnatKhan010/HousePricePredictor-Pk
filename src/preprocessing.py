"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing tasks
    
    Attributes:
        data (pd.DataFrame): Input dataset
        scaler: StandardScaler for normalization
        encoders (Dict): Dictionary of LabelEncoders for categorical columns
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the preprocessor
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy() if data is not None else None
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}")
            logger.info(f"Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, strategy: str = "median") -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        if strategy == "median":
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                    
        elif strategy == "mean":
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                    
        elif strategy == "drop":
            self.data.dropna(inplace=True)
            
        logger.info(f"Missing values handled. New shape: {self.data.shape}")
        return self.data
    
    def detect_outliers(self, column: str, method: str = "iqr") -> pd.DataFrame:
        """
        Detect and remove outliers
        
        Args:
            column (str): Column to check for outliers
            method (str): Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: Dataset without outliers
        """
        logger.info(f"Detecting outliers in '{column}' using {method} method")
        
        if method == "iqr":
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
            
        elif method == "zscore":
            z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
            self.data = self.data[z_scores < 3]
            
        logger.info(f"Outliers removed. New shape: {self.data.shape}")
        return self.data
    
    def encode_categorical(self, categorical_columns: list) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            categorical_columns (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        logger.info(f"Encoding categorical variables: {categorical_columns}")
        
        for col in categorical_columns:
            if col in self.data.columns:
                self.encoders[col] = LabelEncoder()
                self.data[col] = self.encoders[col].fit_transform(self.data[col].astype(str))
                
        logger.info("Categorical variables encoded successfully")
        return self.data
    
    def normalize_features(self, numeric_columns: list = None) -> pd.DataFrame:
        """
        Normalize numeric features
        
        Args:
            numeric_columns (list): List of numeric column names to normalize
            
        Returns:
            pd.DataFrame: Dataset with normalized features
        """
        if numeric_columns is None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
        logger.info(f"Normalizing features: {numeric_columns}")
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])
        logger.info("Features normalized successfully")
        
        return self.data
    
    def feature_engineering(self) -> pd.DataFrame:
        """
        Create new features from existing data
        
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Example: Create interaction features or derived features
        # This can be customized based on domain knowledge
        
        logger.info("Feature engineering completed")
        return self.data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of the data
        
        Returns:
            Dict: Statistics dictionary
        """
        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "describe": self.data.describe().to_dict()
        }


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data("data/raw/zameen_islamabad.csv")
    preprocessor.handle_missing_values(strategy="median")
    preprocessor.detect_outliers(column="price", method="iqr")
