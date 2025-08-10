import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class MarketPatternModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Complete feature preparation pipeline"""
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        df['price_change_1d'] = df['Close'].pct_change(1)
        df['price_change_5d'] = df['Close'].pct_change(5)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        df['future_price'] = df['Close'].shift(-5)
        df['price_change_future'] = (df['future_price'] - df['Close']) / df['Close']
        df['target'] = np.where(df['price_change_future'] > 0.02, 1, 0)
        
        return df.dropna()
    
    def train_model(self, symbol='AAPL', period='1y'):
        filename = f'data/raw/{symbol}_{period}.csv'
        if not os.path.exists(filename):
            print(f"Data file {filename} not found. Please run data collection first.")
            return
        
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        df_features = self.prepare_features(df)
        
        feature_cols = [
            'SMA_20', 'SMA_50', 'MACD', 'MACD_signal', 'MACD_diff',
            'RSI', 'BB_upper', 'BB_lower', 'BB_middle',
            'price_change_1d', 'price_change_5d', 'volume_ratio'
        ]
        
        X = df_features[feature_cols]
        y = df_features['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.feature_columns = feature_cols
        
        y_pred = self.model.predict(X_test_scaled)
        print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self
    
    def save_model(self, filepath='models/market_pattern_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"✅ Model saved to {filepath}")

if __name__ == "__main__":
    symbol = input("Enter the stock ticker symbol for training (e.g., AAPL): ").upper()
    period = input("Enter the time period for training (e.g., 1y, 6mo, 3mo): ")
    model = MarketPatternModel()
    model.train_model(symbol, period)
    model.save_model()