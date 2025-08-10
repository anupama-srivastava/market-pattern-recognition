import pandas as pd
import numpy as np

class FeatureEngineer:
    def prepare_features(self, df):
        """Complete feature preparation pipeline"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Price patterns
        df['price_change_1d'] = df['Close'].pct_change(1)
        df['price_change_5d'] = df['Close'].pct_change(5)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Target variable
        df['future_price'] = df['Close'].shift(-5)
        df['price_change_future'] = (df['future_price'] - df['Close']) / df['Close']
        df['target'] = np.where(df['price_change_future'] > 0.02, 1, 0)
        
        return df.dropna()