"""
Economic Calendar Integration Module

This module integrates economic calendar data to analyze market events and their impact.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

class EconomicCalendarAnalyzer:
    """Class for analyzing economic calendar events"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.finnhub.io/api/v1"  # Finnhub API
        
    def get_economic_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch economic calendar events
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with economic events
        """
        url = f"{self.base_url}/calendar/economic"
        params = {
            'token': self.api_key,
            'from': start_date,
            'to': end_date
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get('economicCalendar', []))
        else:
            print(f"Error fetching economic calendar: {response.status_code}")
            return pd.DataFrame()
    
    def get_earnings_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch earnings calendar
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with earnings events
        """
        url = f"{self.base_url}/calendar/earnings"
        params = {
            'token': self.api_key,
            'from': start_date,
            'to': end_date
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get('earningsCalendar', []))
        else:
            print(f"Error fetching earnings calendar: {response.status_code}")
            return pd.DataFrame()
    
    def analyze_event_impact(self, symbol: str, events_df: pd.DataFrame) -> dict:
        """
        Analyze the impact of economic events on a stock
        
        Args:
            symbol: Stock ticker symbol
            events_df: DataFrame with economic events
            
        Returns:
            Dictionary with impact analysis
        """
        # This is a placeholder for actual impact analysis
        # In practice, you would analyze price movements around events
        
        return {
            'symbol': symbol,
            'total_events': len(events_df),
            'high_impact_events': len(events_df[events_df['impact'] == 'High']),
            'medium_impact_events': len(events_df[events_df['impact'] == 'Medium']),
            'low_impact_events': len(events_df[events_df['impact'] == 'Low'])
        }

# Example usage
if __name__ == "__main__":
    api_key = "YOUR_API_KEY"  # Replace with your actual API key
    analyzer = EconomicCalendarAnalyzer(api_key)
    
    # Get economic calendar for the next week
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    economic_events = analyzer.get_economic_calendar(start_date, end_date)
    earnings_events = analyzer.get_earnings_calendar(start_date, end_date)
    
    if not economic_events.empty:
        print("Economic Events:")
        print(economic_events.head())
    
    if not earnings_events.empty:
        print("\nEarnings Events:")
        print(earnings_events.head())
