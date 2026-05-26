# House Price Predictor - Pakistan 🏠

A professional machine learning project for predicting house prices in Pakistan using regression models. This project implements multiple ML algorithms and provides a comprehensive analysis of housing market data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The **House Price Predictor** project aims to accurately predict residential property prices in Pakistan based on various features such as location, size, number of rooms, amenities, and market conditions. This is a supervised learning regression task that helps stakeholders in the real estate market make data-driven decisions.

### Key Objectives:
- Perform exploratory data analysis (EDA) on housing market data
- Engineer meaningful features from raw data
- Train and evaluate multiple regression models
- Compare model performance and select the best performer
- Provide predictions and insights for real estate valuation

## ✨ Features

- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Automated and manual feature creation and selection
- **Multiple Models**: Implementation of various regression algorithms:
  - Linear Regression
  - Ridge and Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Models
  - Support Vector Regression (SVR)
  - XGBoost
- **Model Evaluation**: Metrics including RMSE, MAE, R² Score, and Cross-Validation
- **Hyperparameter Tuning**: Grid Search and Randomized Search for optimization
- **Visualization**: Comprehensive plots and charts for insights
- **Production Ready**: Modular, clean, and well-documented code

## 📁 Project Structure

```
HousePricePredictor-Pk/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/                           # Original, unmodified data
│   ├── processed/                     # Cleaned and processed data
│   └── external/                      # External datasets (if any)
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # EDA and data exploration
│   ├── 02_feature_engineering.ipynb  # Feature creation and selection
│   └── 03_model_development.ipynb    # Model training and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Data loading utilities
│   │   └── data_processor.py         # Data cleaning and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py         # Feature engineering functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py            # Model training pipeline
│   │   ├── predict_model.py          # Prediction utilities
│   │   └── model_utils.py            # Model evaluation and comparison
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py              # Plotting functions
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                # Helper functions
│
├── configs/
│   └── config.yaml                   # Project configuration
│
├── models/                            # Trained models directory
│   └── .gitkeep
│
├── reports/
│   ├── figures/                      # Generated figures and plots
│   └── results.md                    # Results and findings
│
├── scripts/
│   ├── train.py                      # Main training script
│   └── predict.py                    # Prediction script
│
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py           # Unit tests
│
└── logs/
    └── .gitkeep
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HasnatKhan010/HousePricePredictor-Pk.git
   cd HousePricePredictor-Pk
   ```

2. **Create a virtual environment:**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n house-price-predictor python=3.9
   conda activate house-price-predictor
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Data Preparation
```bash
python scripts/train.py --prepare-data
```

### 2. Train Models
```bash
python scripts/train.py --train-all
```

### 3. Make Predictions
```bash
python scripts/predict.py --input data/processed/test_data.csv
```

### 4. Jupyter Notebooks
```bash
jupyter notebook
# Navigate to notebooks/ folder for detailed analysis
```

## 📊 Data

### Data Sources
- Raw housing market data from Pakistan
- Features include: location, property type, size (Sq. Ft.), number of bedrooms, bathrooms, amenities, and price

### Data Files
- `data/raw/`: Original CSV files
- `data/processed/`: Cleaned and engineered features

### Data Dimensions
- Training samples: [To be updated]
- Features: [To be updated]
- Target variable: Price (PKR)

## 🤖 Models

### Implemented Models:
1. **Linear Regression**: Baseline model for comparison
2. **Ridge & Lasso**: Regularized linear models to prevent overfitting
3. **Random Forest**: Ensemble method capturing non-linear relationships
4. **Gradient Boosting**: Sequential ensemble learning
5. **XGBoost**: Optimized gradient boosting framework
6. **SVR**: Support Vector Regression for complex patterns

### Model Selection Criteria:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)
- Cross-Validation performance

## 📈 Results

### Model Performance Summary
| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| Best Model | - | - | - |

*Results will be updated after model training*

### Key Insights
- [To be populated with findings]
- Feature importance rankings
- Price prediction accuracy

## 🔧 Configuration

Edit `configs/config.yaml` to customize:
- Model hyperparameters
- Data paths
- Train/test split ratio
- Random seed for reproducibility

Example:
```yaml
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  test_size: 0.2
  random_state: 42

models:
  xgboost:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5
```

## 📝 Requirements

See `requirements.txt` for all dependencies. Main libraries:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- xgboost, lightgbm: Advanced boosting models
- matplotlib, seaborn: Data visualization
- jupyter: Interactive notebooks

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Hasnat Khan**
- GitHub: [@HasnatKhan010](https://github.com/HasnatKhan010)
- University: FAST-NUCES
- Student ID: SP24-BCS-039

## 🙏 Acknowledgments

- FAST-NUCES for educational support
- Open-source ML community
- Scikit-learn, XGBoost, and other libraries used

## 📞 Contact

For questions or suggestions, please open an issue or contact me via GitHub.

---

**Last Updated**: May 26, 2026
**Project Status**: 🔄 In Development
