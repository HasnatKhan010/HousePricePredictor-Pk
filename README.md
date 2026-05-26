# 🏡 Islamabad House Price Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-CatBoost%20%7C%20Scikit--Learn-orange)](https://catboost.ai/)
[![Web Scraping](https://img.shields.io/badge/Scraping-Selenium%20%7C%20BeautifulSoup-green)](https://selenium.dev/)
[![GUI](https://img.shields.io/badge/GUI-CustomTkinter-lightblue)](https://github.com/TomSchimansky/CustomTkinter)
[![License](https://img.shields.io/badge/License-MIT-success.svg)](LICENSE)

> **An end-to-end Machine Learning solution for predicting real estate prices in Islamabad using web-scraped data from Zameen.com**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Course Learning Outcomes](#course-learning-outcomes)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

This project implements a complete Machine Learning pipeline to predict residential property prices in Islamabad. It demonstrates end-to-end ML expertise including:
- **Data Engineering**: Web scraping, data cleaning, feature engineering
- **Machine Learning**: Model training, hyperparameter tuning, and evaluation
- **Software Engineering**: Production-ready GUI application with CustomTkinter

---

## 🌟 Key Features

✅ **Automated Data Pipeline**
- Custom Selenium-based web scraper for real-time data collection from Zameen.com
- Intelligent data preprocessing with missing value handling and outlier detection
- Feature engineering with location-based aggregations

✅ **Advanced Machine Learning**
- Gradient Boosting models (CatBoost, XGBoost)
- Traditional models (Linear Regression, Random Forest)
- Comprehensive model evaluation and comparison
- Automated model selection and persistence

✅ **Interactive Desktop Application**
- User-friendly GUI built with CustomTkinter
- Real-time price predictions with feature input validation
- Location-based price estimation
- Intuitive visualization of results

✅ **Production-Ready Code**
- Modular and well-documented codebase
- Error handling and logging
- Reproducible results with fixed random seeds

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────┐
│   Data Collection (Web Scraping)        │
│   - Zameen.com Real Estate Listings     │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Data Preprocessing & EDA              │
│   - Cleaning, Validation, Feature Eng   │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Model Development & Training          │
│   - Multiple ML Algorithms              │
│   - Hyperparameter Optimization         │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Model Evaluation & Selection          │
│   - Cross-validation, Performance Metrics
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   GUI Prediction Application            │
│   - Real-time Price Estimation          │
└─────────────────────────────────────────┘
```

---

## 📊 Dataset

**Dataset**: `zameen_islamabad.csv`
- **Size**: 158 KB | **Records**: 500+ properties
- **Source**: Zameen.com (Pakistan's largest real estate portal)
- **Features**: Location, Area, Bedrooms, Bathrooms, Price, Type, etc.
- **Target Variable**: Price (in PKR)

**Data Characteristics**:
- Coverage: All major areas of Islamabad
- Time Period: Recent listings (continuously updated)
- Data Quality: Cleaned and validated with outlier detection

---

## 🚀 Installation

### Prerequisites
- **Python 3.8+** (Tested with Python 3.9, 3.10, 3.11)
- **pip** (Python package manager)
- **Virtual Environment** (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/HasnatKhan010/HousePricePredictor-Pk.git
cd HousePricePredictor-Pk
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import catboost, customtkinter, selenium; print('✅ All dependencies installed successfully!')"
```

---

## 💻 Usage

### Option 1: Run the GUI Application (Recommended)
Start the interactive desktop application for price predictions:
```bash
python "4 GUI  Prediction System.py"
```

**How to use the GUI**:
1. Enter property details (area, bedrooms, bathrooms, etc.)
2. Select location from dropdown
3. Click "Predict Price"
4. View estimated price in PKR

### Option 2: View ML Pipeline (Jupyter Notebook)
Explore data analysis, preprocessing, and model training:
```bash
jupyter notebook "2 ml_pipeline.ipynb"
```

**Sections covered**:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Comparison
- Hyperparameter Tuning
- Performance Evaluation

### Option 3: Scrape Fresh Data (Advanced)
Generate updated dataset from Zameen.com:
```bash
python "1 Scraper code.py"
```

**Note**: May take 5-10 minutes depending on internet speed.

---

## 📁 Project Structure

```
HousePricePredictor-Pk/
├── 1 Scraper code.py              # Web scraper module
├── 2 ml_pipeline.ipynb            # ML pipeline notebook
├── 4 GUI  Prediction System.py    # GUI application
├── zameen_islamabad.csv           # Dataset
├── best_model.pkl                 # Trained model
├── feature_cols.pkl               # Feature columns
├── global_median.pkl              # Preprocessing data
├── loc_freq_map.pkl               # Location frequency map
├── loc_median_map.pkl             # Location median prices
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

---

## 📈 Model Performance

| Model | R² Score | RMSE (PKR) | MAE (PKR) |
|-------|----------|-----------|----------|
| **CatBoost** | **0.89** | **850K** | **620K** |
| XGBoost | 0.86 | 920K | 680K |
| Random Forest | 0.84 | 980K | 710K |
| Linear Regression | 0.78 | 1.2M | 890K |

**Selected Model**: CatBoost (Best performance)

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Web Scraping** | Selenium, BeautifulSoup4 |
| **Data Processing** | Pandas, NumPy, Scikit-Learn |
| **Machine Learning** | CatBoost, XGBoost, Scikit-Learn |
| **Visualization** | Matplotlib, Seaborn |
| **GUI Framework** | CustomTkinter |
| **Model Persistence** | Pickle |
| **Notebook** | Jupyter |

---

## 🎓 Course Learning Outcomes

This project covers key competencies in:

| CLO | Description | Implementation |
|-----|-------------|-----------------|
| **CLO-1** | Data acquisition and web scraping | `1 Scraper code.py` |
| **CLO-2** | Data preprocessing and feature engineering | `2 ml_pipeline.ipynb` (Section 2-3) |
| **CLO-3** | Regression model development | `2 ml_pipeline.ipynb` (Section 4-5) |
| **CLO-4** | Model evaluation and comparison | `2 ml_pipeline.ipynb` (Section 6) |
| **CLO-5** | Production system implementation | `4 GUI  Prediction System.py` |

---

## 📝 Dependencies

See `requirements.txt` for complete list:

```
catboost==1.2.2
xgboost==2.0.3
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
selenium==4.13.0
beautifulsoup4==4.12.2
customtkinter==5.2.0
jupyter==1.0.0
matplotlib==3.8.1
seaborn==0.13.0
requests==2.31.0
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Zameen.com** for providing real estate data
- **CustomTkinter** for the modern GUI framework
- **CatBoost team** for the excellent gradient boosting library

---

**Last Updated**: May 2026 | **Status**: Complete ✅
