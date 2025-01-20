import ssl
import os
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from fredapi import Fred
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.config.run_functions_eagerly(True)
ssl._create_default_https_context = ssl._create_unverified_context

# QVKYT4BJ7903SJXV

def fetch_stock_data(ticker):
    """Fetch stock data from Alpha Vantage API"""
    api_key = 'QVKYT4BJ7903SJXV'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full'
    r = requests.get(url)
    data = r.json()
    time_series = data.get('Time Series (Daily)', {})
    df = pd.DataFrame.from_dict(time_series, orient='index', dtype=float)
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume',
    })
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# b21ce6567289b6d5c0d8d6d998f0b4da

def fetch_fred_data(stock_data):
    """Fetch macroeconomic data aligned with stock data date range."""
    fred = Fred(api_key='b21ce6567289b6d5c0d8d6d998f0b4da')
    # Expand the date range to ensure sufficient overlap
    start_date = stock_data.index.min() - pd.DateOffset(years=1)
    end_date = stock_data.index.max() + pd.DateOffset(years=1)

    interest_rates = fred.get_series('FEDFUNDS', observation_start=start_date, observation_end=end_date)
    gdp_growth = fred.get_series('A191RL1Q225SBEA', observation_start=start_date, observation_end=end_date)
    inflation = fred.get_series('CPIAUCSL', observation_start=start_date, observation_end=end_date)

    df_macro = pd.DataFrame({
        'Interest_Rates': interest_rates,
        'GDP_Growth': gdp_growth,
        'Inflation': inflation
    })
    df_macro.index = pd.to_datetime(df_macro.index)
    df_macro.sort_index(inplace=True)
    df_macro.fillna(method='ffill', inplace=True)
    return df_macro

def fetch_company_overview(ticker, api_key):
    """Fetch financial data for the company using Alpha Vantage Company Overview API."""
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    if 'Symbol' not in data:
        raise ValueError("Invalid API response. Company Overview data is missing.")

    def safe_float(value, default=0.0):
        """Safely convert a value to float, defaulting to 0.0 if conversion fails."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # Extract key financial metrics, handling missing or None values
    financial_data = {
        "MarketCapitalization": safe_float(data.get("MarketCapitalization")),
        "PE_Ratio": safe_float(data.get("PERatio")),
        "DividendYield": safe_float(data.get("DividendYield")),
        "EPS": safe_float(data.get("EPS")),
        "ProfitMargin": safe_float(data.get("ProfitMargin")),
    }
    return financial_data

def fetch_market_sentiment(ticker, api_key):
    """Fetch market news sentiment for a specific ticker."""
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    sentiment_scores = []
    if 'feed' in data:
        for article in data['feed']:
            # Ensure relevance to the ticker
            if ticker.lower() in article.get('ticker', '').lower():
                sentiment_scores.append(float(article.get('overall_sentiment_score', 0)))

    # Calculate average sentiment for the ticker
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

def preprocess_data(df, df_macro):
    """Preprocess stock data with additional features."""
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Bollinger_Upper'] = df['MA_10'] + 2 * df['Close'].rolling(window=10).std()
    df['Bollinger_Lower'] = df['MA_10'] - 2 * df['Close'].rolling(window=10).std()
    df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Price_Change'] = df['Close'].pct_change()
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    df['Month'] = df.index.month
    df['Returns'] = df['Close'].pct_change()
    df['Lagged_Return'] = df['Returns'].shift(1)
    df = df.merge(df_macro, left_index=True, right_index=True, how='left')
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing data
    df.fillna(method='bfill', inplace=True)  # Backward-fill if necessary
    df.dropna(inplace=True)  # Finally, drop rows with critical missing values
    return df

def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest with GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestRegressor(random_state=42)
    # Use min(3, len(X_train)) for safe cross-validation
    n_splits = min(3, len(X_train))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.ravel())
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model"""
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mse, r2

def prepare_forecast_features(df, forecast_days, user_interest_rate, user_gdp_growth, user_inflation, financial_data, sentiment_score):
    """Prepare features for future price forecasting with rolling updates."""
    last_row = df.iloc[-1].copy()  # Fetch the last row of the DataFrame

    # Initialize rolling averages for RSI
    last_row['Avg_Gain'] = 0 if 'Avg_Gain' not in last_row else last_row['Avg_Gain']
    last_row['Avg_Loss'] = 0 if 'Avg_Loss' not in last_row else last_row['Avg_Loss']

    # Required features for forecasting
    required_features = [
        'Close_Lag1', 'Close_Lag2', 'MA_5', 'MA_10', 'EMA_12', 'EMA_26',
        'MACD', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR',
        'Volatility', 'Price_Change', 'Volume', 'Interest_Rates',
        'GDP_Growth', 'Inflation', 'MarketCapitalization', 'PE_Ratio',
        'DividendYield', 'EPS', 'ProfitMargin', 'Sentiment_Score'
    ]
    forecast_features = []

    for _ in range(forecast_days):
        # Populate all required features
        features = {feature: last_row.get(feature, 0) for feature in required_features}
        features.update({
            'Interest_Rates': user_interest_rate,
            'GDP_Growth': user_gdp_growth,
            'Inflation': user_inflation,
            'MarketCapitalization': financial_data['MarketCapitalization'],
            'PE_Ratio': financial_data['PE_Ratio'],
            'DividendYield': financial_data['DividendYield'],
            'EPS': financial_data['EPS'],
            'ProfitMargin': financial_data['ProfitMargin'],
            'Sentiment_Score': sentiment_score
        })

        # Simulate rolling forward
        next_close = features['Close_Lag1'] * (1 + np.random.uniform(-0.01, 0.01))  # Simulate slight change
        delta = next_close - last_row['Close_Lag1']  # Difference between forecasted and previous close

        # Update RSI
        gain = max(delta, 0)
        loss = abs(min(delta, 0))
        avg_gain = (last_row['Avg_Gain'] * 13 + gain) / 14  # Rolling average for gains
        avg_loss = (last_row['Avg_Loss'] * 13 + loss) / 14  # Rolling average for losses
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        last_row['RSI'] = 100 - (100 / (1 + rs))
        last_row['Avg_Gain'] = avg_gain
        last_row['Avg_Loss'] = avg_loss

        # Update Price Change
        last_row['Price_Change'] = (next_close / last_row['Close_Lag1']) - 1

        # Update volume variability
        last_row['Volume'] = last_row['Volume'] * (1 + np.random.uniform(-0.01, 0.01))  

        # Update rolling features
        last_row['Close_Lag2'] = last_row['Close_Lag1']
        last_row['Close_Lag1'] = next_close
        last_row['MA_5'] = (last_row['MA_5'] * 4 + next_close) / 5 if 'MA_5' in last_row else next_close
        last_row['MA_10'] = (last_row['MA_10'] * 9 + next_close) / 10 if 'MA_10' in last_row else next_close
        last_row['EMA_12'] = (next_close * (2 / (12 + 1))) + (last_row['EMA_12'] * (1 - (2 / (12 + 1))))
        last_row['EMA_26'] = (next_close * (2 / (26 + 1))) + (last_row['EMA_26'] * (1 - (2 / (26 + 1))))
        last_row['MACD'] = last_row['EMA_12'] - last_row['EMA_26']
        last_row['Bollinger_Upper'] = last_row['MA_10'] + 2 * last_row['Volatility']
        last_row['Bollinger_Lower'] = last_row['MA_10'] - 2 * last_row['Volatility']
        last_row['Volatility'] = last_row['Volatility'] * 0.9 + abs(next_close - last_row['Close_Lag1']) * 0.1  # Smoothing

        # Append current features to forecast features
        forecast_features.append(features)

    return pd.DataFrame(forecast_features)

# Streamlit UI
st.title("Stock Price Prediction and Visualization")
ticker = st.text_input("Enter the Stock Ticker Symbol (e.g., AAPL):", "AAPL")
assert isinstance(ticker, str), f"Expected ticker to be a string, got {type(ticker)}"

# Sidebar inputs for user-defined macroeconomic indicators
st.sidebar.title("Enter desired rates and forecast period")
if 'user_interest_rate' not in st.session_state:
    st.session_state['user_interest_rate'] = 5.0
if 'user_gdp_growth' not in st.session_state:
    st.session_state['user_gdp_growth'] = 3.0
if 'user_inflation' not in st.session_state:
    st.session_state['user_inflation'] = 2.0
if 'forecast_days' not in st.session_state:
    st.session_state['forecast_days'] = 30

user_interest_rate = st.sidebar.number_input(
    "Interest Rate (e.g., 5.0 for 5%)",
    value=st.session_state['user_interest_rate'],
    step=0.1,
    key="interest_rate_input"
)
user_gdp_growth = st.sidebar.number_input(
    "GDP Growth (e.g., 3.0 for 3%)",
    value=st.session_state['user_gdp_growth'],
    step=0.1,
    key="gdp_growth_input"
)
user_inflation = st.sidebar.number_input(
    "Inflation Rate (e.g., 2.0 for 2%)",
    value=st.session_state['user_inflation'],
    step=0.1,
    key="inflation_rate_input"
)
forecast_days = st.sidebar.number_input(
    "Number of Days to Forecast",
    value=st.session_state['forecast_days'],
    step=1,
    key="forecast_days_input"
)

# Update session state
st.session_state['user_interest_rate'] = user_interest_rate
st.session_state['user_gdp_growth'] = user_gdp_growth
st.session_state['user_inflation'] = user_inflation

if st.button("Fetch and Analyze Data"):
    try:
        # Main Program
        df_stock = fetch_stock_data(ticker)

        # Fetch macroeconomic data
        df_macro = fetch_fred_data(df_stock)

        # Preprocess data
        df = preprocess_data(df_stock, df_macro)

        # Fetch company overview data
        company_financials = fetch_company_overview(ticker, api_key = "QVKYT4BJ7903SJXV")

        # Fetch sentiment score
        sentiment_score = fetch_market_sentiment(ticker, api_key = "QVKYT4BJ7903SJXV")

        if len(df) < 50:  # Set a reasonable minimum
            st.error("Insufficient data for training. Try expanding the date range.")
            raise ValueError("Insufficient data for model training.")

        features = [
            'Close_Lag1', 'Close_Lag2', 'MA_5', 'MA_10', 'EMA_12', 'EMA_26',
            'MACD', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR',
            'Volatility', 'Price_Change', 'Volume', 'Interest_Rates',
            'GDP_Growth', 'Inflation', 'MarketCapitalization', 'PE_Ratio',
            'DividendYield', 'EPS', 'ProfitMargin', 'Sentiment_Score'
        ]

        # Add placeholder columns to X for alignment with forecast features
        for col in ['MarketCapitalization', 'PE_Ratio', 'DividendYield', 'EPS', 'ProfitMargin', 'Sentiment_Score']:
            if col not in df.columns:
                df[col] = 0

        target = 'Close'
        X = df[features]
        y = df['Close']

        # Scale the features
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(X)

        # Scale the target variable (y) separately
        target_scaler = MinMaxScaler()  # or RobustScaler()
        y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(scaled_data, y_scaled, test_size=0.3, random_state=42, shuffle=False)

        # Train Random Forest
        rf_model = optimize_random_forest(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        st.session_state['rf_model'] = rf_model

        # Train XGBoost
        xgb_model, xgb_pred, xgb_mse, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test)
        st.session_state['xgb_model'] = xgb_model

        # Filter the test data for the last year
        test_start_index = int(len(df) * 0.7)  # Assuming a 70-30 train-test split
        test_index = df.index[test_start_index:]  # Indices corresponding to the test set

        last_year_date = pd.to_datetime(test_index[-1]) - pd.DateOffset(years=1)
        last_year_indices = test_index > last_year_date

        # Apply the filter to the test data
        test_index_last_year = test_index[last_year_indices]
        y_test_last_year = y_test[last_year_indices]
        rf_pred_last_year = rf_pred[last_year_indices]
        xgb_pred_last_year = xgb_pred[last_year_indices]

        # Reverse scaling for actual and predictions
        y_test_actual = target_scaler.inverse_transform(y_test_last_year.reshape(-1, 1))
        rf_pred_actual = target_scaler.inverse_transform(rf_pred_last_year.reshape(-1, 1))
        xgb_pred_actual = target_scaler.inverse_transform(xgb_pred_last_year.reshape(-1, 1))

        # Ensure all predictions and actual values have the same length
        common_length = min(len(y_test_actual), len(rf_pred_actual), len(xgb_pred_actual))
        y_test_actual = y_test_actual[-common_length:]
        rf_pred_actual = rf_pred_actual[-common_length:]
        xgb_pred_actual = xgb_pred_actual[-common_length:]
        aligned_index = df.index[-common_length:]  # Align index to match

        # Plot the predictions and actual values
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_index_last_year, y_test_actual, label='Actual Prices', color='blue')
        ax.plot(test_index_last_year, rf_pred_actual, label='Random Forest Predictions', color='orange')
        ax.plot(test_index_last_year, xgb_pred_actual, label='XGBoost Predictions', color='green')
        # plt.plot(aligned_index, lstm_pred_actual, label='LSTM Predictions', color='red')
        ax.set_title('Actual vs Predicted Stock Prices (Random Forest, XGBoost)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)

        # FORECAST FUTURE PRICES - Automatically triggered after the plot
        st.write("Generating future price forecasts...")

        # Prepare forecast features using values from st.session_state
        forecast_features = prepare_forecast_features(
            df=df,
            forecast_days=forecast_days,
            user_interest_rate=user_interest_rate,
            user_gdp_growth=user_gdp_growth,
            user_inflation=user_inflation,
            financial_data=company_financials,
            sentiment_score=sentiment_score
        )

        # Debugging - Checking number of features are equal
        # st.write("Forecast Features (before alignment):", forecast_features.columns)
        # st.write("Training Features:", X.columns)

        # Debugging - Checking shape of arrays are equal    
        # st.write("Forecast Features Shape:", forecast_features.shape)
        # st.write("Training Features Shape:", X.shape)

        # Align forecast_features with training features
        forecast_features = forecast_features[X.columns]

        # Scale forecast features
        forecast_scaled = scaler.transform(forecast_features)

        # Predict using trained models
        rf_forecast = st.session_state['rf_model'].predict(forecast_scaled)
        xgb_forecast = st.session_state['xgb_model'].predict(forecast_scaled)

        # Reverse scale predictions
        rf_forecast_actual = target_scaler.inverse_transform(rf_forecast.reshape(-1, 1))
        xgb_forecast_actual = target_scaler.inverse_transform(xgb_forecast.reshape(-1, 1))

        # Create forecast DataFrame
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=st.session_state['forecast_days'])
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'RF_Forecast': rf_forecast_actual.flatten(),
            'XGB_Forecast': xgb_forecast_actual.flatten()
        })

        # # Debugging for alignment
        # st.write("Forecast Features for Debugging:", forecast_features.head(10))
        # st.write("Forecast Features:", forecast_features.columns)
        # st.write("Training Features:", X.columns)
        # st.write("Mismatch:", set(forecast_features.columns) - set(X.columns))

        # Plot forecasted prices only
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot forecasted prices
        if 'RF_Forecast' in forecast_df and 'XGB_Forecast' in forecast_df:
            ax.plot(forecast_df['Date'], forecast_df['RF_Forecast'], label='RF Forecast', color='orange')
            ax.plot(forecast_df['Date'], forecast_df['XGB_Forecast'], label='XGB Forecast', color='green')
        else:
            st.error("Forecast data is missing or not properly computed.")

        # Set plot labels and title
        ax.set_title('Forecasted Stock Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(f"An error occurred: {e}")
        st.text("Traceback:")
        st.text(traceback.format_exc())

