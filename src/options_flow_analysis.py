"""
Options Flow Analysis Module

This module analyzes unusual options activity, implied volatility,
and options flow to identify potential trading opportunities.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptionsFlowAnalyzer:
    """Advanced options flow analysis for trading signals"""
    
    def __init__(self):
        self.base_url = "https://api.polygon.io"
        
    def get_options_chain(self, symbol: str, expiration_date: str) -> pd.DataFrame:
        """
        Get options chain data for a given symbol and expiration
        
        Args:
            symbol: Stock ticker symbol
            expiration_date: Options expiration date
            
        Returns:
            DataFrame with options chain data
        """
        ticker = yf.Ticker(symbol)
        
        try:
            # Get options chain
            options = ticker.option_chain(expiration_date)
            
            # Combine calls and puts
            calls = options.calls.copy()
            puts = options.puts.copy()
            
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            
            # Combine and add symbol
            options_chain = pd.concat([calls, puts])
            options_chain['symbol'] = symbol
            
            return options_chain
            
        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_implied_volatility_rank(self, symbol: str) -> Dict:
        """
        Calculate implied volatility rank and percentile
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with IV rank and percentile
        """
        ticker = yf.Ticker(symbol)
        
        try:
            # Get historical volatility data
            hist_data = ticker.history(period="1y")
            returns = hist_data['Close'].pct_change().dropna()
            
            # Calculate historical volatility
            hist_vol = returns.rolling(window=30).std() * np.sqrt(252)
            current_vol = hist_vol.iloc[-1]
            
            # Get options implied volatility
            expirations = ticker.options
            if not expirations:
                return {}
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            options_chain = self.get_options_chain(symbol, nearest_exp)
            
            if options_chain.empty:
                return {}
            
            # Calculate average implied volatility
            avg_iv = options_chain['impliedVolatility'].mean()
            
            # Calculate IV rank
            iv_rank = ((avg_iv - hist_vol.min()) / (hist_vol.max() - hist_vol.min())) * 100
            
            return {
                'current_iv': avg_iv,
                'historical_vol': current_vol,
                'iv_rank': iv_rank,
                'iv_percentile': (hist_vol <= avg_iv).mean() * 100
            }
            
        except Exception as e:
            print(f"Error calculating IV rank for {symbol}: {e}")
            return {}
    
    def detect_unusual_options_activity(self, symbol: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect unusual options activity based on volume vs open interest
        
        Args:
            symbol: Stock ticker symbol
            threshold: Volume/OI threshold for unusual activity
            
        Returns:
            DataFrame with unusual options activity
        """
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return pd.DataFrame()
            
            unusual_activity = []
            
            for expiration in expirations[:3]:  # Check next 3 expirations
                options_chain = self.get_options_chain(symbol, expiration)
                
                if options_chain.empty:
                    continue
                
                # Calculate volume/OI ratio
                options_chain['volume_oi_ratio'] = options_chain['volume'] / (options_chain['openInterest'] + 1)
                
                # Filter for unusual activity
                unusual = options_chain[options_chain['volume_oi_ratio'] > threshold]
                
                if not unusual.empty:
                    unusual['expiration'] = expiration
                    unusual_activity.append(unusual)
            
            if unusual_activity:
                return pd.concat(unusual_activity)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error detecting unusual activity for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_options_flow(self, symbol: str) -> Dict:
        """
        Analyze options flow for bullish/bearish sentiment
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with options flow analysis
        """
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return {}
            
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            
            # Analyze next 3 expirations
            for expiration in expirations[:3]:
                options_chain = self.get_options_chain(symbol, expiration)
                
                if options_chain.empty:
                    continue
                
                calls = options_chain[options_chain['option_type'] == 'call']
                puts = options_chain[options_chain['option_type'] == 'put']
                
                total_call_volume += calls['volume'].sum()
                total_put_volume += puts['volume'].sum()
                total_call_oi += calls['openInterest'].sum()
                total_put_oi += puts['openInterest'].sum()
            
            # Calculate ratios
            put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Detect large block trades
            unusual_activity = self.detect_unusual_options_activity(symbol)
            
            return {
                'symbol': symbol,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'put_call_ratio': put_call_ratio,
                'put_call_oi_ratio': put_call_oi_ratio,
                'unusual_activity': unusual_activity,
                'bullish_sentiment': put_call_ratio < 0.7,
                'bearish_sentiment': put_call_ratio > 1.3
            }
            
        except Exception as e:
            print(f"Error analyzing options flow for {symbol}: {e}")
            return {}
    
    def get_options_sweep_data(self, symbol: str) -> pd.DataFrame:
        """
        Get options sweep data (large block trades)
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with sweep data
        """
        # This would typically use a paid API like Cheddar Flow, FlowAlgo, or Unusual Whales
        # For demonstration, we'll simulate sweep detection
        
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return pd.DataFrame()
            
            sweep_data = []
            
            for expiration in expirations[:2]:
                options_chain = self.get_options_chain(symbol, expiration)
                
                if options_chain.empty:
                    continue
                
                # Identify large trades (simulated)
                large_trades = options_chain[
                    (options_chain['volume'] > 1000) & 
                    (options_chain['volume'] > options_chain['openInterest'] * 0.5)
                ]
                
                if not large_trades.empty:
                    large_trades['expiration'] = expiration
                    large_trades['trade_type'] = 'sweep'
                    sweep_data.append(large_trades)
            
            if sweep_data:
                return pd.concat(sweep_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting sweep data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_options_signals(self, symbol: str) -> Dict:
        """
        Generate trading signals based on options flow analysis
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with trading signals
        """
        analysis = self.analyze_options_flow(symbol)
        
        if not analysis:
            return {}
        
        signals = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'put_call_ratio': analysis['put_call_ratio'],
            'bullish_signal': analysis['bullish_sentiment'],
            'bearish_signal': analysis['bearish_sentiment'],
            'unusual_activity': not analysis['unusual_activity'].empty,
            'iv_rank': self.calculate_implied_volatility_rank(symbol).get('iv_rank', 0)
        }
        
        # Generate composite signal
        if signals['bullish_signal'] and signals['unusual_activity']:
            signals['recommendation'] = 'BULLISH'
        elif signals['bearish_signal'] and signals['unusual_activity']:
            signals['recommendation'] = 'BEARISH'
        else:
            signals['recommendation'] = 'NEUTRAL'
        
        return signals

# Example usage
if __name__ == "__main__":
    analyzer = OptionsFlowAnalyzer()
    
    # Analyze options flow for popular stocks
    symbols = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'META']
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        signals = analyzer.generate_options_signals(symbol)
        
        if signals:
            print(f"Recommendation: {signals['recommendation']}")
            print(f"Put/Call Ratio: {signals['put_call_ratio']:.2f}")
            print(f"Unusual Activity: {signals['unusual_activity']}")
            print(f"IV Rank: {signals['iv_rank']:.1f}%")
