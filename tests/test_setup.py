try:
    import yfinance as yf
    import pandas as pd
    import sklearn
    print("✅ All dependencies installed successfully")
    
    # Test data download
    aapl = yf.Ticker("AAPL")
    data = aapl.history(period="5d")
    print(f"✅ Data download working - got {len(data)} rows")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
except Exception as e:
    print(f"❌ Error: {e}")