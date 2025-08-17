"""
Advanced Feature Engineering Module
Provides comprehensive technical indicators and market analysis features
"""

import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedTechnicalIndicators:
    """Comprehensive technical analysis indicators for market prediction"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available technical indicators"""
        df = df.copy()
        
        # Basic price features
        df = self._add_basic_features(df)
        
        # Moving averages
        df = self._add_moving_averages(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Support/Resistance levels
        df = self._add_support_resistance(df)
        
        # Ichimoku Cloud
        df = self._add_ichimoku_cloud(df)
        
        # Fibonacci retracements
        df = self._add_fibonacci_levels(df)
        
        # Market microstructure
        df = self._add_market_microstructure(df)
        
        # Regime detection
        df = self._add_regime_indicators(df)
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_range'] = df['High'] - df['Low']
        df['price_range_pct'] = df['price_range'] / df['Close']
        df['close_to_high'] = (df['High'] - df['Close']) / df['price_range']
        df['close_to_low'] = (df['Close'] - df['Low']) / df['price_range']
        
        # Price position in range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            # Simple Moving Averages
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
            df[f'WMA_{period}'] = ta.trend.WMAIndicator(df['Close'], window=period).wma()
            df[f'HMA_{period}'] = self._calculate_hma(df['Close'], period)
            
            # Moving average relationships
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'SMA_{period}']
            df[f'sma_{period}_slope'] = df[f'SMA_{period}'].diff()
            df[f'sma_{period}_acceleration'] = df[f'sma_{period}_slope'].diff()
        
        # Golden Cross/Death Cross signals
        df['golden_cross'] = ((df['SMA_50'] > df['SMA_200']) & 
                             (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))).astype(int)
        df['death_cross'] = ((df['SMA_50'] < df['SMA_200']) & 
                            (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))).astype(int)
        
        return df
    
    def _calculate_hma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        
        wma_half = ta.trend.WMAIndicator(series, window=half_length).wma()
        wma_full = ta.trend.WMAIndicator(series, window=period).wma()
        
        raw_hma = 2 * wma_half - wma_full
        hma = ta.trend.WMAIndicator(raw_hma, window=sqrt_length).wma()
        
        return hma
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        
        # RSI variants
        rsi_periods = [6, 14, 21, 28]
        for period in rsi_periods:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
            df[f'rsi_{period}_slope'] = df[f'rsi_{period}'].diff()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).williams_r()
        
        # Rate of Change
        roc_periods = [5, 10, 20]
        for period in roc_periods:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['Close'], window=period).roc()
        
        # Momentum
        momentum_periods = [10, 20]
        for period in momentum_periods:
            df[f'momentum_{period}'] = ta.momentum.momentum_indicator(
                df['Close'], window=period
            )
        
        # Ultimate Oscillator
        df['ultimate_oscillator'] = ta.momentum.UltimateOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).ultimate_oscillator()
        
        # Awesome Oscillator
        df['awesome_oscillator'] = ta.momentum.AwesomeOscillatorIndicator(
            high=df['High'],
            low=df['Low']
        ).awesome_oscillator()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        
        # Bollinger Bands
        bb_periods = [20, 50]
        for period in bb_periods:
            bb = ta.volatility.BollingerBands(df['Close'], window=period)
            df[f'bb_upper_{period}'] = bb.bollinger_hband()
            df[f'bb_lower_{period}'] = bb.bollinger_lband()
            df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            df[f'bb_width_{period}'] = bb.bollinger_wband()
            df[f'bb_percent_{period}'] = bb.bollinger_pband()
        
        # Average True Range
        atr_periods = [14, 21]
        for period in atr_periods:
            df[f'atr_{period}'] = ta.volatility.AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=period
            ).average_true_range()
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_middle'] = kc.keltner_channel_mband()
        
        # Donchian Channel
        dc = ta.volatility.DonchianChannel(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
        df['dc_upper'] = dc.donchian_channel_hband()
        df['dc_lower'] = dc.donchian_channel_lband()
        df['dc_middle'] = dc.donchian_channel_mband()
        
        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # Volume Moving Averages
        volume_periods = [10, 20, 50]
        for period in volume_periods:
            df[f'volume_sma_{period}'] = ta.volume.VolumeSMAIndicator(
                df['Volume'], window=period
            ).volume_sma()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
        
        # On Balance Volume
        df['on_balance_volume'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'],
            volume=df['Volume']
        ).on_balance_volume()
        
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Chaikin Money Flow
        df['chaikin_money_flow'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).chaikin_money_flow()
        
        # Force Index
        force_index = ta.volume.ForceIndexIndicator(
            close=df['Close'],
            volume=df['Volume']
        )
        df['force_index_13'] = force_index.force_index()
        
        # Money Flow Index
        df['money_flow_index'] = ta.volume.MFIIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).money_flow_index()
        
        # Accumulation/Distribution Line
        df['acc_dist_line'] = ta.volume.AccDistIndexIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).acc_dist_index()
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels"""
        
        periods = [10, 20, 50]
        
        for period in periods:
            # Support levels
            df[f'support_{period}'] = df['Low'].rolling(window=period).min()
            df[f'resistance_{period}'] = df['High'].rolling(window=period).max()
            
            # Distance from support/resistance
            df[f'distance_from_support_{period}'] = (df['Close'] - df[f'support_{period}']) / df[f'support_{period}']
            df[f'distance_from_resistance_{period}'] = (df[f'resistance_{period}'] - df['Close']) / df[f'resistance_{period}']
            
            # Breakout signals
            df[f'support_break_{period}'] = (df['Close'] < df[f'support_{period}']).astype(int)
            df[f'resistance_break_{period}'] = (df['Close'] > df[f'resistance_{period}']).astype(int)
        
        return df
    
    def _add_ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators"""
        
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['High'],
            low=df['Low'],
            window1=9,
            window2=26,
            window3=52
        )
        
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # Cloud signals
        df['above_cloud'] = (df['Close'] > df['ichimoku_a']) & (df['Close'] > df['ichimoku_b'])
        df['below_cloud'] = (df['Close'] < df['ichimoku_a']) & (df['Close'] < df['ichimoku_b'])
        df['in_cloud'] = ~df['above_cloud'] & ~df['below_cloud']
        
        return df
    
    def _add_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement levels"""
        
        lookback = 50
        
        # Calculate high and low over lookback period
        high_50 = df['High'].rolling(window=lookback).max()
        low_50 = df['Low'].rolling(window=lookback).min()
        range_50 = high_50 - low_50
        
        # Fibonacci levels
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        for level in fib_levels:
            df[f'fib_{int(level*1000)}'] = low_50 + (range_50 * level)
        
        # Current position relative to Fibonacci levels
        for level in fib_levels:
            df[f'fib_{int(level*1000)}_distance'] = (df['Close'] - df[f'fib_{int(level*1000)}']) / df[f'fib_{int(level*1000)}']
        
        return df
    
    def _add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Order flow imbalance (simplified)
        df['order_flow_imbalance'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Price impact
        df['price_impact'] = (df['Close'] - df['Open']) / df['Volume']
        
        # Spread measures
        df['spread'] = df['High'] - df['Low']
        df['spread_pct'] = df['spread'] / df['Close']
        
        # Volume profile
        df['volume_profile'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        return df
    
    def _add_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection indicators"""
        
        # Volatility regime
        df['volatility_regime'] = pd.cut(
            df['returns'].rolling(window=20).std(),
            bins=3,
            labels=['low', 'medium', 'high']
        )
        
        # Trend regime using moving averages
        df['trend_regime'] = 0
        df.loc[df['Close'] > df['SMA_50'], 'trend_regime'] = 1
        df.loc[df['Close'] < df['SMA_50'], 'trend_regime'] = -1
        
        # Momentum regime
        df['momentum_regime'] = pd.cut(
            df['rsi_14'],
            bins=[0, 30, 70, 100],
            labels=['oversold', 'neutral', 'overbought']
        )
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate feature importance using correlation with returns"""
        
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        correlations = {}
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                corr = df[col].corr(df['returns'])
                correlations[col] = abs(corr) if not pd.isna(corr) else 0
        
        return pd.Series(correlations).sort_values(ascending=False)

class FeatureSelector:
    """Feature selection and dimensionality reduction"""
    
    def __init__(self):
        self.selected_features = []
        
    def select_features(self, df: pd.DataFrame, target_col: str = 'returns', 
                       threshold: float = 0.1) -> List[str]:
        """Select most relevant features based on correlation"""
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        correlations = {}
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                corr = df[col].corr(df[target_col])
                if abs(corr) > threshold and not pd.isna(corr):
                    correlations[col] = abs(corr)
        
        self.selected_features = list(correlations.keys())
        return self.selected_features
    
    def create_feature_matrix(self, df: pd.DataFrame, 
                            selected_features: List[str] = None) -> pd.DataFrame:
        """Create feature matrix with selected features"""
        
        if selected_features is None:
            selected_features = self.selected_features
            
        return df[selected_features].fillna(method='ffill').fillna(0)

# Usage example
if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    data = yf.download("AAPL", period="1y")
    
    # Initialize feature engineer
    feature_engineer = AdvancedTechnicalIndicators()
    
    # Calculate all indicators
    df_enhanced = feature_engineer.calculate_all_indicators(data)
    
    # Select important features
    selector = FeatureSelector()
    important_features = selector.select_features(df_enhanced)
    
    print(f"✅ Enhanced features calculated: {len(df_enhanced.columns)} features")
    print(f"✅ Selected features: {len(important_features)} features")
    
    # Save enhanced data
    df_enhanced.to_csv('data/processed/enhanced_features.csv')
