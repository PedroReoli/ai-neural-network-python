import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import pandas_ta as pta

class DataFetcher:
    def __init__(self, symbol='AAPL', start_date=None, end_date=None):
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance"""
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            return self._preprocess_data(data)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def _preprocess_data(self, data):
        """Preprocess the data for model training"""
        # Basic price data
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving Averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Volatility
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['ATR'] = ta.volatility.average_true_range(
            data['High'], data['Low'], data['Close']
        )
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['Stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
        
        # Volume Indicators
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['VWAP'] = ta.volume.volume_weighted_average_price(
            data['High'], data['Low'], data['Close'], data['Volume']
        )
        
        # Trend Indicators
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
        
        # Bollinger Bands
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.volatility.bollinger_bands(data['Close'])
        
        # Fibonacci Retracement Levels
        data['Fib_0.236'] = data['Close'].rolling(window=20).max() * 0.236
        data['Fib_0.382'] = data['Close'].rolling(window=20).max() * 0.382
        data['Fib_0.618'] = data['Close'].rolling(window=20).max() * 0.618
        
        # Target variables
        data['Target_Direction'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        data['Target_Intensity'] = abs(data['Returns'].shift(-1))
        
        # Additional features
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Price_Change'] = data['Close'].pct_change()
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def get_training_data(self):
        """Get processed data ready for model training"""
        data = self.fetch_data()
        if data is None:
            return None
            
        features = [
            'Returns', 'Log_Returns', 'MA5', 'MA20', 'MA50', 'MA200',
            'Volatility', 'ATR', 'RSI', 'MACD', 'Stoch', 'OBV', 'VWAP',
            'ADX', 'CCI', 'Price_Range', 'Volume_Change', 'Price_Change'
        ]
        
        X = data[features].values
        y_direction = data['Target_Direction'].values
        y_intensity = data['Target_Intensity'].values
        
        return X, y_direction, y_intensity
    
    def get_market_summary(self):
        """Get a summary of market conditions"""
        data = self.fetch_data()
        if data is None:
            return None
            
        summary = {
            'current_price': data['Close'].iloc[-1],
            'daily_change': data['Returns'].iloc[-1] * 100,
            'volatility': data['Volatility'].iloc[-1] * 100,
            'rsi': data['RSI'].iloc[-1],
            'trend': 'Bullish' if data['MA5'].iloc[-1] > data['MA20'].iloc[-1] else 'Bearish',
            'volume_trend': 'Increasing' if data['Volume_Change'].iloc[-1] > 0 else 'Decreasing',
            'support_level': data['BB_lower'].iloc[-1],
            'resistance_level': data['BB_upper'].iloc[-1]
        }
        
        return summary 