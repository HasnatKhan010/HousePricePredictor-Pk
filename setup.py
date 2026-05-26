"""
Setup configuration for House Price Predictor package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="house-price-predictor",
    version="1.0.0",
    author="Developer",
    description="ML pipeline for predicting house prices in Islamabad",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HasnatKhan010/HousePricePredictor-Pk",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "catboost>=1.2.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "selenium>=4.13.0",
        "beautifulsoup4>=4.12.0",
        "customtkinter>=5.2.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "house-predictor-gui=src.gui_app:main",
        ],
    },
)
