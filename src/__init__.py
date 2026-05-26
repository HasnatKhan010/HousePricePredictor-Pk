"""
House Price Predictor Package
ML pipeline for predicting real estate prices in Islamabad
"""

__version__ = "1.0.0"
__author__ = "Developer"
__description__ = "ML pipeline for predicting house prices in Islamabad using web-scraped data from Zameen.com"

from src.scraper import PropertyScraper
from src.preprocessing import DataPreprocessor
from src.model import PricePredictor

__all__ = [
    "PropertyScraper",
    "DataPreprocessor", 
    "PricePredictor",
]
