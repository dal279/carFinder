# Car Price Estimator

**By:** Lukas Chang (ldc105), Daniel Li (dal279)

## Overview

The Car Price Estimator is a tool for estimating the price of used cars based on user inputs, recommending similar cars, and predicting depreciation trends. It integrates data preprocessing, machine learning, and a user-friendly GUI for streamlined decision-making.

## Files

### `car_price_estimator.py`
The main application providing a Tkinter GUI for price estimation, similar car recommendations, and depreciation predictions. Uses SQLite for database operations and a regression model for price predictions.

### `csv_condenser.py`
A preprocessing script that cleans, reduces, and balances the dataset by sampling 70% recent and 30% older records. Outputs a cleaned CSV and metadata for use in the main application.

### `csv_analyzer.py`
A utility script for analyzing datasets, providing insights such as record counts and date ranges to validate input files.

## How to Run `car_price_estimator.py`

### Prerequisites
Ensure you have the following dependencies installed:

- Python 3.8+
- Required Python libraries:
  ```bash
  pip3 install pandas numpy matplotlib scikit-learn sqlite3 hashlib tkinter

## Running the App
	1.	Run car_price_estimator.py:
            python3 car_price_estimator.py
    2.	The app will launch a GUI where you can:
        - Select Car Details: Use the dropdowns and input fields to provide details about the car (e.g., manufacturer, model, year, mileage, fuel type, and transmission).
        - Estimate Price: Click the “Estimate Price” button to calculate the car’s estimated market value.
        - View Similar Cars: After estimating the price, the app will display a list of similar cars with their details in the output area, including mileage and price.
        - Predict Depreciation: Click the “Predict Depreciation” button to generate an interactive graph showing the car’s depreciation over the next 10 years.
