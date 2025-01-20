import requests
import pandas as pd

def fetch_stock_data():
    """Fetch stock data from Alpha Vantage API"""
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=your_api_key'
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
    print(data)
    return df
fetch_stock_data()

