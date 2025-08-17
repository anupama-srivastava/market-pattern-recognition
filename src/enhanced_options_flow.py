"""
Enhanced Options Flow Analysis Module

This module provides advanced options flow analysis with real-time data integration.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple
import asyncio
import aiohttp

class EnhancedOptionsFlowAnalyzer:
    """Enhanced options flow analysis with advanced features"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        
    async def get_options_chain_async(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Async version of options chain retrieval"""
        url = f"https://api.polygon.io/v2/aggs/ticker/O:{symbol}{expiration}/range/1/minute/2023-01-09/2023-01-09"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return pd.DataFrame(data.get('results', []))
                return pd.DataFrame()
    
    def calculate_greeks(self, options_data: pd.DataFrame, 
                        underlying_price: float, 
                        risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Calculate option Greeks using Black-Scholes model
        
        Args:
            options_data: DataFrame with options data
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame with Greeks added
        """
        from scipy.stats import norm
        
        def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
            """Calculate Black-Scholes Greeks"""
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
                vega = S * norm.pdf(d1) * np.sqrt(T)
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:
                delta = -norm.cdf(-d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
                vega = S * norm.pdf(d1) * np.sqrt(T)
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
                
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        
        # Calculate Greeks for each option
        greeks_data = []
        for _, row in options_data.iterrows():
            greeks = black_scholes_greeks(
                S=underlying_price,
                K=row['strike'],
                T=row['days_to_expiry'] / 365,
                r=risk_free_rate,
                sigma=row['implied_volatility'],
                option_type=row['option_type']
            )
            greeks_data.append(greeks)
        
        greeks_df = pd.DataFrame(greeks_data)
        return pd.concat([options_data, greeks_df], axis=1)
    
    def detect_unusual_activity(self, symbol: str, 
                              volume_threshold: float = 3.0,
                              oi_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect unusual options activity with enhanced criteria
        
        Args:
            symbol: Stock ticker symbol
            volume_threshold: Volume multiplier threshold
            oi_threshold: Open interest multiplier threshold
            
        Returns:
            DataFrame with unusual activity
        """
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return pd.DataFrame()
            
            unusual_activity = []
            
            for expiration in expirations[:5]:  # Check next 5 expirations
                options = ticker.option_chain(expiration)
                
                # Process calls and puts
                calls = options.calls.copy()
                puts = options.puts.copy()
                
                calls['option_type'] = 'call'
                puts['option_type'] = 'put'
                
                all_options = pd.concat([calls, puts])
                all_options['expiration'] = expiration
                
                # Calculate unusual metrics
                all_options['volume_oi_ratio'] = all_options['volume'] / (all_options['openInterest'] + 1)
                all_options['volume_avg_ratio'] = all_options['volume'] / all_options['volume'].mean()
                
                # Filter unusual activity
                unusual = all_options[
                    (all_options['volume_oi_ratio'] > volume_threshold) |
                    (all_options['volume_avg_ratio'] > oi_threshold) |
                    (all_options['volume'] > 1000)
                ]
                
                if not unusual.empty:
                    unusual_activity.append(unusual)
            
            if unusual_activity:
                return pd.concat(unusual_activity)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error detecting unusual activity: {e}")
            return pd.DataFrame()
    
    def analyze_options_flow_sentiment(self, symbol: str) -> Dict:
        """
        Analyze options flow sentiment with advanced metrics
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment analysis
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
            
            # Analyze all expirations
            for expiration in expirations:
                options = ticker.option_chain(expiration)
                
                calls = options.calls
                puts = options.puts
                
                total_call_volume += calls['volume'].sum()
                total_put_volume += puts['volume'].sum()
                total_call_oi += calls['openInterest'].sum()
                total_put_oi += puts['openInterest'].sum()
            
            # Calculate sentiment metrics
            put_call_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Detect unusual activity
            unusual_activity = self.detect_unusual_activity(symbol)
            
            # Calculate sentiment score
            sentiment_score = 0
            if put_call_volume_ratio < 0.7:
                sentiment_score = 1  # Bullish
            elif put_call_volume_ratio > 1.3:
                sentiment_score = -1  # Bearish
            
            return {
                'symbol': symbol,
                'put_call_volume_ratio': put_call_volume_ratio,
                'put_call_oi_ratio': put_call_oi_ratio,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'sentiment_score': sentiment_score,
                'unusual_activity_count': len(unusual_activity),
                'bullish_sentiment': sentiment_score > 0,
                'bearish_sentiment': sentiment_score < 0
            }
            
        except Exception as e:
            print(f"Error analyzing options flow: {e}")
            return {}
    
    def get_sweep_data(self, symbol: str) -> pd.DataFrame:
        """
        Get options sweep data (large institutional trades)
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with sweep data
        """
        # This would typically use a paid API
        # For demonstration, we'll simulate sweep detection
        
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return pd.DataFrame()
            
            sweep_data = []
            
            for expiration in expirations[:3]:
                options = ticker.option_chain(expiration)
                
                # Combine calls and puts
                calls = options.calls
                puts = options.puts
                
                calls['option_type'] = 'call'
                puts['option_type'] = 'put'
                
                all_options = pd.concat([calls, puts])
                
                # Identify sweeps
                sweeps = all_options[
                    (all_options['volume'] > 500) &
                    (all_options['volume'] > all_options['openInterest'] * 0.5) &
                    (all_options['volume'] > all_options['volume'].quantile(0.95))
                ]
                
                if not sweeps.empty:
                    sweeps['expiration'] = expiration
                    sweeps['sweep_type'] = 'large_block'
                    sweep_data.append(sweeps)
            
            if sweep_data:
                return pd.concat(sweep_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting sweep data: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    analyzer = EnhancedOptionsFlowAnalyzer()
    
    # Analyze options flow for popular stocks
    symbols = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'META']
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        sentiment = analyzer.analyze_options_flow_sentiment(symbol)
        
        if sentiment:
            print(f"Sentiment Score: {sentiment['sentiment_score']}")
            print(f"Put/Call Volume Ratio: {sentiment['put_call_volume_ratio']:.2f}")
            print(f"Unusual Activity: {sentiment['unusual_activity_count']}")
