"""
Advanced Candlestick Pattern Recognition Module
Implements both traditional pattern detection and computer vision-based pattern recognition
"""

import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class TraditionalPatternDetector:
    """Traditional candlestick pattern detection using price data"""
    
    def __init__(self):
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'hanging_man': self._detect_hanging_man,
            'engulfing': self._detect_engulfing,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'shooting_star': self._detect_shooting_star,
            'inverted_hammer': self._detect_inverted_hammer,
            'dark_cloud_cover': self._detect_dark_cloud_cover,
            'piercing_line': self._detect_piercing_line,
            'three_white_soldiers': self._detect_three_white_soldiers,
            'three_black_crows': self._detect_three_black_crows
        }
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all traditional candlestick patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern detection results
        """
        df = df.copy()
        
        # Calculate basic candlestick metrics
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['total_range'] = df['High'] - df['Low']
        
        # Calculate relative sizes
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
        
        # Detect patterns
        for pattern_name, pattern_func in self.patterns.items():
            df[pattern_name] = pattern_func(df)
        
        return df
    
    def _detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """Detect Doji patterns (open ≈ close)"""
        return (df['body_ratio'] < 0.1) & (df['total_range'] > 0)
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer patterns (small body, long lower shadow)"""
        return (
            (df['body_ratio'] < 0.3) &
            (df['lower_shadow_ratio'] > 2 * df['body_ratio']) &
            (df['upper_shadow_ratio'] < 0.1)
        )
    
    def _detect_hanging_man(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hanging Man patterns (hammer at top of uptrend)"""
        hammer = self._detect_hammer(df)
        uptrend = df['Close'] > df['Close'].shift(5).rolling(5).mean()
        return hammer & uptrend
    
    def _detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect Bullish/Bearish Engulfing patterns"""
        bullish_engulfing = (
            (df['Close'] > df['Open']) &  # Current candle is bullish
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous candle is bearish
            (df['Close'] > df['Open'].shift(1)) &  # Current close > previous open
            (df['Open'] < df['Close'].shift(1))  # Current open < previous close
        )
        
        bearish_engulfing = (
            (df['Close'] < df['Open']) &  # Current candle is bearish
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous candle is bullish
            (df['Close'] < df['Open'].shift(1)) &  # Current close < previous open
            (df['Open'] > df['Close'].shift(1))  # Current open > previous close
        )
        
        return bullish_engulfing.astype(int) - bearish_engulfing.astype(int)
    
    def _detect_morning_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Morning Star patterns (bullish reversal)"""
        # First candle: long bearish
        first_bearish = (df['Close'].shift(2) < df['Open'].shift(2)) & \
                       (abs(df['Close'].shift(2) - df['Open'].shift(2)) > df['body_size'].rolling(20).mean().shift(2))
        
        # Second candle: small body (star)
        second_small = abs(df['Close'].shift(1) - df['Open'].shift(1)) < df['body_size'].rolling(20).mean().shift(1) * 0.5
        
        # Third candle: long bullish
        third_bullish = (df['Close'] > df['Open']) & \
                       (abs(df['Close'] - df['Open']) > df['body_size'].rolling(20).mean())
        
        return first_bearish & second_small & third_bullish
    
    def _detect_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Evening Star patterns (bearish reversal)"""
        # First candle: long bullish
        first_bullish = (df['Close'].shift(2) > df['Open'].shift(2)) & \
                       (abs(df['Close'].shift(2) - df['Open'].shift(2)) > df['body_size'].rolling(20).mean().shift(2))
        
        # Second candle: small body (star)
        second_small = abs(df['Close'].shift(1) - df['Open'].shift(1)) < df['body_size'].rolling(20).mean().shift(1) * 0.5
        
        # Third candle: long bearish
        third_bearish = (df['Close'] < df['Open']) & \
                       (abs(df['Close'] - df['Open']) > df['body_size'].rolling(20).mean())
        
        return first_bullish & second_small & third_bearish
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Shooting Star patterns (bearish reversal)"""
        return (
            (df['Close'] < df['Open']) &  # Bearish candle
            (df['upper_shadow_ratio'] > 2 * df['body_ratio']) &
            (df['lower_shadow_ratio'] < 0.1)
        )
    
    def _detect_inverted_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Inverted Hammer patterns (bullish reversal)"""
        return (
            (df['Close'] > df['Open']) &  # Bullish candle
            (df['upper_shadow_ratio'] > 2 * df['body_ratio']) &
            (df['lower_shadow_ratio'] < 0.1)
        )
    
    def _detect_dark_cloud_cover(self, df: pd.DataFrame) -> pd.Series:
        """Detect Dark Cloud Cover patterns (bearish reversal)"""
        first_bullish = (df['Close'].shift(1) > df['Open'].shift(1))
        second_bearish = (df['Close'] < df['Open'])
        price_condition = (df['Close'] < df['Open'].shift(1)) & (df['Close'] > df['Close'].shift(1))
        
        return first_bullish & second_bearish & price_condition
    
    def _detect_piercing_line(self, df: pd.DataFrame) -> pd.Series:
        """Detect Piercing Line patterns (bullish reversal)"""
        first_bearish = (df['Close'].shift(1) < df['Open'].shift(1))
        second_bullish = (df['Close'] > df['Open'])
        price_condition = (df['Close'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
        
        return first_bearish & second_bullish & price_condition
    
    def _detect_three_white_soldiers(self, df: pd.DataFrame) -> pd.Series:
        """Detect Three White Soldiers patterns (strong bullish)"""
        cond1 = (df['Close'] > df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(2) > df['Open'].shift(2))
        cond2 = (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2))
        cond3 = (df['Open'] > df['Open'].shift(1)) & (df['Open'].shift(1) > df['Open'].shift(2))
        
        return cond1 & cond2 & cond3
    
    def _detect_three_black_crows(self, df: pd.DataFrame) -> pd.Series:
        """Detect Three Black Crows patterns (strong bearish)"""
        cond1 = (df['Close'] < df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(2) < df['Open'].shift(2))
        cond2 = (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2))
        cond3 = (df['Open'] < df['Open'].shift(1)) & (df['Open'].shift(1) < df['Open'].shift(2))
        
        return cond1 & cond2 & cond3

class ComputerVisionPatternDetector:
    """Computer vision-based pattern detection using image processing"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
    def create_candlestick_image(self, df: pd.DataFrame, window_size: int = 50) -> np.ndarray:
        """
        Create candlestick chart image from OHLC data
        
        Args:
            df: DataFrame with OHLC data
            window_size: Number of candles to include in image
            
        Returns:
            Image array suitable for CV processing
        """
        # Select recent data
        recent_data = df.tail(window_size)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot candlesticks
        for idx, row in recent_data.iterrows():
            x = range(len(recent_data)).index(idx)
            
            # Body
            if row['Close'] > row['Open']:
                color = 'green'
                bottom = row['Open']
                height = row['Close'] - row['Open']
            else:
                color = 'red'
                bottom = row['Close']
                height = row['Open'] - row['Close']
            
            # Plot body
            ax.bar(x, height, bottom=bottom, width=0.8, color=color, alpha=0.8)
            
            # Plot wicks
            ax.plot([x, x], [row['Low'], row['High']], color='black', linewidth=1)
        
        # Format plot
        ax.set_xlim(-1, len(recent_data))
        ax.set_ylim(recent_data['Low'].min() * 0.99, recent_data['High'].max() * 1.01)
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        plt.close(fig)
        
        return image
    
    def detect_support_resistance(self, df: pd.DataFrame, window_size: int = 50) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels using computer vision
        
        Args:
            df: DataFrame with OHLC data
            window_size: Number of data points to analyze
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Get closing prices
        prices = df['Close'].tail(window_size).values
        
        # Find peaks (resistance) and troughs (support)
        peaks, _ = find_peaks(prices, distance=5, prominence=np.std(prices) * 0.5)
        troughs, _ = find_peaks(-prices, distance=5, prominence=np.std(prices) * 0.5)
        
        # Get price levels
        resistance_levels = prices[peaks].tolist()
        support_levels = prices[troughs].tolist()
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'support_indices': troughs.tolist(),
            'resistance_indices': peaks.tolist()
        }
    
    def detect_trend_lines(self, df: pd.DataFrame, window_size: int = 50) -> Dict[str, List[Tuple[float, float]]]:
        """
        Detect trend lines using computer vision techniques
        
        Args:
            df: DataFrame with OHLC data
            window_size: Number of data points to analyze
            
        Returns:
            Dictionary with trend line equations
        """
        # Get price data
        prices = df['Close'].tail(window_size).values
        x = np.arange(len(prices))
        
        # Detect trend lines using linear regression
        from sklearn.linear_model import LinearRegression
        
        # Uptrend line (connecting lows)
        lows = df['Low'].tail(window_size).values
        uptrend_model = LinearRegression()
        
        # Find best uptrend line
        best_uptrend_r2 = 0
        best_uptrend_line = None
        
        for i in range(len(lows) - 5):
            for j in range(i + 5, len(lows)):
                x_line = x[i:j+1].reshape(-1, 1)
                y_line = lows[i:j+1].reshape(-1, 1)
                
                uptrend_model.fit(x_line, y_line)
                r2 = uptrend_model.score(x_line, y_line)
                
                if r2 > best_uptrend_r2:
                    best_uptrend_r2 = r2
                    slope = uptrend_model.coef_[0][0]
                    intercept = uptrend_model.intercept_[0]
                    best_uptrend_line = (slope, intercept)
        
        # Downtrend line (connecting highs)
        highs = df['High'].tail(window_size).values
        downtrend_model = LinearRegression()
        
        best_downtrend_r2 = 0
        best_downtrend_line = None
        
        for i in range(len(highs) - 5):
            for j in range(i + 5, len(highs)):
                x_line = x[i:j+1].reshape(-1, 1)
                y_line = highs[i:j+1].reshape(-1, 1)
                
                downtrend_model.fit(x_line, y_line)
                r2 = downtrend_model.score(x_line, y_line)
                
                if r2 > best_downtrend_r2:
                    best_downtrend_r2 = r2
                    slope = downtrend_model.coef_[0][0]
                    intercept = downtrend_model.intercept_[0]
                    best_downtrend_line = (slope, intercept)
        
        return {
            'uptrend_line': best_uptrend_line,
            'downtrend_line': best_downtrend_line,
            'uptrend_r2': best_uptrend_r2,
            'downtrend_r2': best_downtrend_r2
        }

class PatternFeatureEngineer:
    """Integrate pattern detection into feature engineering pipeline"""
    
    def __init__(self):
        self.traditional_detector = TraditionalPatternDetector()
        self.cv_detector = ComputerVisionPatternDetector()
    
    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive pattern-based features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern features
        """
        df = df.copy()
        
        # Detect traditional patterns
        df_patterns = self.traditional_detector.detect_patterns(df)
        
        # Create pattern-based features
        pattern_cols = [col for col in df_patterns.columns if col in self.traditional_detector.patterns.keys()]
        
        # Pattern counts in rolling windows
        for window in [5, 10, 20]:
            for pattern in pattern_cols:
                df[f'{pattern}_count_{window}'] = df[pattern].rolling(window).sum()
                df[f'{pattern}_ratio_{window}'] = df[pattern].rolling(window).mean()
        
        # Pattern confidence scores
        df['pattern_diversity'] = df[pattern_cols].sum(axis=1)
        df['bullish_patterns'] = df[[p for p in pattern_cols if 'bullish' in p.lower()]].sum(axis=1)
        df['bearish_patterns'] = df[[p for p in pattern_cols if 'bearish' in p.lower()]].sum(axis=1)
        
        # Pattern momentum
        df['pattern_momentum'] = df['bullish_patterns'] - df['bearish_patterns']
        
        # CV-based features
        try:
            cv_features = self.cv_detector.detect_support_resistance(df.tail(100))
            df['support_levels_count'] = len(cv_features['support_levels'])
            df['resistance_levels_count'] = len(cv_features['resistance_levels'])
        except:
            df['support_levels_count'] = 0
            df['resistance_levels_count'] = 0
        
        return df

# Example usage
if __name__ == "__main__":
    # Create sample data
    import yfinance as yf
    
    # Get data
    data = yf.download("AAPL", period="3mo", interval="1d")
    
    # Initialize pattern detector
    pattern_engineer = PatternFeatureEngineer()
    
    # Create pattern features
    pattern_features = pattern_engineer.create_pattern_features(data)
    
    print("✅ Pattern features created successfully!")
    print(f"Features shape: {pattern_features.shape}")
    print(f"Pattern columns: {[col for col in pattern_features.columns if 'pattern' in col.lower()]}")
