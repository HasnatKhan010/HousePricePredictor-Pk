# 🏡 Islamabad House Price Prediction

![Python Setup](https://img.shields.io/badge/python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CatBoost%20%7C%20Scikit--Learn-orange)
![Web Scraping](https://img.shields.io/badge/Web%20Scraping-Selenium%20%7C%20BS4-green)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-lightgrey)

An end-to-end Machine Learning project to scrape, process, and predict real estate properties in Islamabad using property listings scraped from Zameen.com. Developed as a final project for **Course: AIC354 - Machine Learning Fundamentals Lab**.

## 🌟 Key Features
- **Data Acquisition**: Custom undetected web scraper to extract real housing data.
- **Robust ML Pipeline**: Automated data cleaning, feature engineering, handling of missing values, and outlier detection.
- **Advanced Modeling**: Utilization of gradient boosting algorithms (CatBoost, XGBoost) alongside traditional Scikit-Learn models.
- **Interactive GUI**: A sleek, user-friendly desktop application built with `CustomTkinter` to input house features and get instant price estimates.

---

## 📂 Project Structure

| File | Description |
|---|---|
| `1 Scraper code.py` | Web scraper built with Selenium and BeautifulSoup to collect raw data. |
| `2 ml_pipeline.ipynb` | Jupyter Notebook containing data preprocessing, EDA, model training, and evaluation. |
| `4 GUI Prediction System.py`| Desktop application (CustomTkinter) for users to predict real estate prices. |
| `zameen_islamabad.csv` | The gathered dataset containing details about properties in Islamabad. |
| `requirements.txt` | Python dependencies required for the ML notebook. |
| `GUI_requirements.txt` | Minimal Python dependencies for running the GUI seamlessly. |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment so your system dependencies do not conflict.

### 2. Installation
Open your terminal and create a virtual environment in the project directory:
```bash
python -m venv .venv
```
Activate the virtual environment:
- **Windows**: `.venv\Scripts\activate`

Install the required dependencies:
```bash
pip install -r requirements.txt
pip install -r GUI_requirements.txt
```

### 3. Usage

#### Run the Web Scraper (Optional)
To scrape the latest property data and generate a new `zameen_islamabad.csv`:
```bash
python "1 Scraper code.py"
```

#### Run the ML Pipeline (Notebook)
To view data analytics, preprocessing steps, and model evaluations:
```bash
jupyter notebook "2 ml_pipeline.ipynb"
```

#### Launch the GUI Application
To run the standalone desktop app that predicts estimated house prices:
```bash
python "4 GUI  Prediction System.py"
```

---

## 🎓 CLO (Course Learning Outcomes) Coverage
- **CLO-1**: Data acquisition and scraping (`1 Scraper code.py`).
- **CLO-2**: Data preprocessing and feature engineering (`2 ml_pipeline.ipynb`).
- **CLO-3**: Regression model development (`2 ml_pipeline.ipynb`).
- **CLO-4**: Model evaluation and comparison (`2 ml_pipeline.ipynb`).
- **CLO-5**: Final prediction system implementation (`4 GUI Prediction System.py`).
