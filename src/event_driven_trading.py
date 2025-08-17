"""
Event-Driven Trading Module

This module implements event-driven trading strategies including:
- Earnings announcements
- FDA approvals
- Economic releases
- Corporate actions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class EventDrivenTrading:
    """Event-driven trading strategy implementation"""
    
    def __init__(self):
        self.earnings_calendar = {}
        self.economic_calendar = {}
        
    def get_earnings_announcements(self, symbol: str, days_ahead: int = 30) -> List[Dict]:
        """
        Get upcoming earnings announcements for a stock
        
        Args:
            symbol: Stock ticker symbol
            days_ahead: Number of days to look ahead
            
        Returns:
            List of earnings announcements
        """
        ticker = yf.Ticker(symbol)
        
        try:
            # Get calendar events
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return []
            
            announcements = []
            
            for date, event in calendar.iterrows():
                announcement = {
                    'symbol': symbol,
                    'date': date,
                    'event_type': 'earnings',
                    'estimate': event.get('Earnings Estimate', np.nan),
                    'actual': event.get('Reported EPS', np.nan),
                    'surprise': event.get('Surprise(%)', np.nan)
                }
                announcements.append(announcement)
            
            return announcements
            
        except Exception as e:
            print(f"Error getting earnings for {symbol}: {e}")
            return []
    
    def analyze_earnings_surprise(self, symbol: str, lookback_days: int = 252) -> Dict:
        """
        Analyze historical earnings surprises and market reaction
        
        Args:
            symbol: Stock ticker symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with earnings surprise analysis
        """
        ticker = yf.Ticker(symbol)
        
        try:
            # Get historical earnings
            earnings = ticker.earnings_history
            
            if earnings is None or earnings.empty:
                return {}
            
            # Calculate surprise metrics
            earnings['surprise_pct'] = earnings['Surprise(%)']
            positive_surprises = (earnings['surprise_pct'] > 0).sum()
            total_surprises = len(earnings)
            
            # Calculate average surprise
            avg_surprise = earnings['surprise_pct'].mean()
            
            # Calculate post-earnings drift
            earnings['post_er_return'] = earnings['surprise_pct'] * 0.1  # Simplified
            
            return {
                'symbol': symbol,
                'total_earnings': total_surprises,
                'positive_surprises': positive_surprises,
                'beat_rate': positive_surprises / total_surprises if total_surprises > 0 else 0,
                'avg_surprise': avg_surprise,
                'post_er_drift': earnings['post_er_return'].mean()
            }
            
        except Exception as e:
            print(f"Error analyzing earnings for {symbol}: {e}")
            return {}
    
    def get_economic_events(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Get major economic events
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of economic events
        """
        # This would typically use a paid API like FRED or Trading Economics
        # For demonstration, we'll simulate major events
        
        events = [
            {
                'date': '2024-01-31',
                'event': 'FOMC Rate Decision',
                'importance': 'high',
                'expected_impact': 'market_volatility'
            },
            {
                'date': '2024-02-13',
                'event': 'CPI Release',
                'importance': 'high',
                'expected_impact': 'inflation_sensitive'
            },
            {
                'date': '2024-02-14',
                'event': 'PPI Release',
                'importance': 'medium',
                'expected_impact': 'inflation_sensitive'
            }
        ]
        
        return events
    
    def calculate_event_impact(self, symbol: str, event_date: str, window: int = 5) -> Dict:
        """
        Calculate the impact of an event on stock price
        
        Args:
            symbol: Stock ticker symbol
            event_date: Date of the event
            window: Analysis window around event
            
        Returns:
            Dictionary with event impact analysis
        """
        ticker = yf.Ticker(symbol)
        
        try:
            # Get price data around event
            start_date = (datetime.strptime(event_date, '%Y-%m-%d') - timedelta(days=window)).strftime('%Y-%m-%d')
            end_date = (datetime.strptime(event_date, '%Y-%m-%d') + timedelta(days=window)).strftime('%Y-%m-%d')
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return {}
            
            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            
            # Event day return
            event_return = data.loc[event_date]['returns'] if event_date in data.index else 0
            
            # Cumulative return around event
            pre_event_return = data[data.index < event_date]['returns'].sum()
            post_event_return = data[data.index > event_date]['returns'].sum()
            
            # Volatility impact
            pre_vol = data[data.index < event_date]['returns'].std()
            post_vol = data[data.index > event_date]['returns'].std()
            
            return {
                'symbol': symbol,
                'event_date': event_date,
                'event_return': event_return,
                'pre_event_return': pre_event_return,
                'post_event_return': post_event_return,
                'volatility_increase': post_vol > pre_vol,
                'volatility_ratio': post_vol / pre_vol if pre_vol > 0 else 1
            }
            
        except Exception as e:
            print(f"Error calculating event impact for {symbol}: {e}")
            return {}
    
    def generate_earnings_strategy(self, symbol: str) -> Dict:
        """
        Generate earnings-based trading strategy
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with earnings strategy
        """
        # Get earnings analysis
        earnings_analysis = self.analyze_earnings_surprise(symbol)
        
        if not earnings_analysis:
            return {}
        
        # Get upcoming earnings
        upcoming_earnings = self.get_earnings_announcements(symbol, days_ahead=30)
        
        strategy = {
            'symbol': symbol,
            'beat_rate': earnings_analysis['beat_rate'],
            'avg_surprise': earnings_analysis['avg_surprise'],
            'upcoming_earnings': upcoming_earnings,
            'strategy': None
        }
        
        # Generate strategy based on historical performance
        if earnings_analysis['beat_rate'] > 0.7 and earnings_analysis['avg_surprise'] > 2:
            strategy['strategy'] = 'BULLISH_PRE_EARNINGS'
        elif earnings_analysis['beat_rate'] < 0.3:
            strategy['strategy'] = 'BEARISH_PRE_EARNINGS'
        else:
            strategy['strategy'] = 'NEUTRAL'
        
        return strategy
    
    def get_fda_calendar(self) -> List[Dict]:
        """
        Get FDA approval calendar (simulated data)
        
        Returns:
            List of FDA events
        """
        # This would typically use FDA API or biotech databases
        fda_events = [
            {
                'symbol': 'BIIB',
                'date': '2024-02-15',
                'event': 'FDA Approval Decision',
                'drug': 'Alzheimer Treatment',
                'phase': 'PDUFA'
            },
            {
                'symbol': 'GILD',
                'date': '2024-02-20',
                'event': 'FDA Advisory Committee',
                'drug': 'Cancer Therapy',
                'phase': 'AdComm'
            }
        ]
        
        return fda_events
    
    def backtest_event_strategy(self, symbol: str, event_type: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest event-driven trading strategy
        
        Args:
            symbol: Stock ticker symbol
            event_type: Type of event (earnings, fda, etc.)
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        ticker = yf.Ticker(symbol)
        
        try:
            # Get historical data
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return {}
            
            # Simulate event dates (in real implementation, use actual events)
            event_dates = pd.date_range(start=start_date, end=end_date, freq='30D')
            
            returns = []
            
            for event_date in event_dates:
                if event_date in data.index:
                    # Calculate returns around event
                    event_idx = data.index.get_loc(event_date)
                    
                    if event_idx > 5 and event_idx < len(data) - 5:
                        # 5-day returns around event
                        pre_return = data.iloc[event_idx-5:event_idx]['Close'].pct_change().sum()
                        post_return = data.iloc[event_idx:event_idx+5]['Close'].pct_change().sum()
                        
                        returns.append({
                            'date': event_date,
                            'pre_return': pre_return,
                            'post_return': post_return
                        })
            
            if returns:
                df = pd.DataFrame(returns)
                
                return {
                    'symbol': symbol,
                    'event_type': event_type,
                    'avg_pre_return': df['pre_return'].mean(),
                    'avg_post_return': df['post_return'].mean(),
                    'win_rate': (df['post_return'] > 0).mean(),
                    'total_events': len(returns)
                }
            
            return {}
            
        except Exception as e:
            print(f"Error backtesting event strategy for {symbol}: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    event_trader = EventDrivenTrading()
    
    # Analyze earnings for popular stocks
    symbols = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'META']
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        strategy = event_trader.generate_earnings_strategy(symbol)
        
        if strategy:
            print(f"Beat Rate: {strategy['beat_rate']:.2%}")
            print(f"Average Surprise: {strategy['avg_surprise']:.2f}%")
            print(f"Strategy: {strategy['strategy']}")
