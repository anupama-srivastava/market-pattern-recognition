import yfinance as yf
import pandas as pd
import os

class StockDataCollector:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_stock_data(self, symbol, period="1y"):
        """Download stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            filename = f"{symbol}_{period}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            print(f"✅ Downloaded {len(df)} rows for {symbol} with period {period}")
            return df
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            return None

if __name__ == "__main__":
    symbol = input("Enter the stock ticker symbol (e.g., AAPL): ").upper()
    period = input("Enter the time period (e.g., 1y, 6mo, 3mo): ")
    collector = StockDataCollector()
    collector.download_stock_data(symbol, period)