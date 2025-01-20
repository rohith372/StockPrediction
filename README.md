# Stock Prediction

## Overview
This program predicts stock prices using machine learning models (Random Forest, XGBoost) by combining historical stock data, macroeconomic indicators, financial metrics, and sentiment analysis. It features interactive visualizations for feature importance and sentiment trends to enhance user understanding of predictions.

## Features
- Fetches historical stock price data using Alpha Vantage's TIME_SERIES_DAILY API.
- Fetches macroeconomic data (e.g., interest rates, GDP growth, inflation) using the FRED API.
- Analyzes sentiment from market news using Alpha Vantage's Alpha Intelligence Market News API.
- Incorporates company financial metrics via Alpha Vantage's Fundamental Data Company Overview API.
- Forecasts future stock prices using Random Forest and XGBoost models.

## Setup Instructions
### Prerequisites
1. Python 3.7 or later: Ensure Python is installed on your system.
2. Dependencies:
   1. Install required Python libraries:
   2. pip install pandas numpy matplotlib scikit-learn tensorflow streamlit fredapi requests xgboost

## Getting API Keys
###  You will need API keys for the following services:
#### Alpha Vantage:
1. Sign up for a free account at Alpha Vantage (https://www.alphavantage.co/documentation/).
2. Copy your API key.
#### FRED API:
1. Register for an account at FRED (https://fred.stlouisfed.org).
2. Generate your API key.
## Inserting API Keys
1. Alpha Vantage: Replace YOUR_ALPHA_VANTAGE_API_KEY in the code with your actual API key
   - Line 28
   - Line 300
   - Line 303
2. FRED: Replace YOUR_FRED_API_KEY in the code with your FRED API key
   - Line 49
## Program Details
### Data Sources
- Stock Data: Alpha Vantage's TIME_SERIES_DAILY endpoint provides daily stock prices, including open, high, low, close, and volume.
- Macroeconomic Indicators: The FRED API retrieves interest rates, GDP growth, and inflation data.
- News Sentiment: Alpha Vantage's Alpha Intelligence Market News provides sentiment scores for the latest news articles.
- Company Financials: Alpha Vantage's Fundamental Data Company Overview API provides financial metrics like market capitalization, PE ratio, dividend yield, and more.
### Key Functionality
#### Data Preprocessing:
- Historical stock prices are preprocessed with calculated features like moving averages, RSI, MACD, and Bollinger Bands.
- Macroeconomic and financial metrics are merged with historical data for training.
#### Forecasting:
- Features for forecasting include user-defined interest rates, GDP growth, inflation, sentiment score, and financial metrics.
- The program rolls forward predictions for a user-specified number of days.
## Important Note
Using the free tier of Alpha Vantage will allow you to call their API 25 times a day. If you are getting an error in your code and you are not sure why it is happening, it is likely that you exceeded that number. Run test.py in your terminal, and it will return a message saying you exceeded your limit. You do not have to wait 24 hours again to run the code.
## Acknowledgments
- Alpha Vantage for stock and sentiment data.
- FRED API for macroeconomic indicators.
