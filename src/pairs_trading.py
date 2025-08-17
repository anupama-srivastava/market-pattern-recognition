"""
Pairs Trading Strategy Module

This module implements statistical arbitrage strategies for pairs trading,
including cointegration testing, spread analysis, and trading signals.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PairsTradingStrategy:
    """Advanced pairs trading implementation with statistical arbitrage"""
    
    def __init__(self, lookback_period: int = 252, z_entry_threshold: float = 2.0, z_exit_threshold: float = 0.5):
        self.lookback_period = lookback_period
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.pairs_data = {}
        
    def find_cointegrated_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs from a universe of stocks
        
        Args:
            prices: DataFrame with price data for multiple stocks
            
        Returns:
            List of tuples (stock1, stock2, p-value) for cointegrated pairs
        """
        n = prices.shape[1]
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = prices.columns[i]
                stock2 = prices.columns[j]
                
                # Test for cointegration
                score, p_value, _ = coint(prices[stock1], prices[stock2])
                
                if p_value < 0.05:  # 5% significance level
                    pairs.append((stock1, stock2, p_value))
        
        # Sort by p-value (most significant first)
        pairs.sort(key=lambda x: x[2])
        return pairs
    
    def calculate_spread(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Calculate the spread between two price series using OLS regression
        
        Args:
            series1: Price series for stock 1
            series2: Price series for stock 2
            
        Returns:
            Spread series
        """
        # Ensure we're working with the same index
        common_index = series1.index.intersection(series2.index)
        x = series1.loc[common_index]
        y = series2.loc[common_index]
        
        # Add constant for regression
        x = sm.add_constant(x)
        
        # Run regression
        model = OLS(y, x).fit()
        hedge_ratio = model.params[1]
        
        # Calculate spread
        spread = y - hedge_ratio * series1
        
        return spread
    
    def calculate_z_score(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate z-score of the spread
        
        Args:
            spread: Spread series
            window: Rolling window for mean and std calculation
            
        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score = (spread - rolling_mean) / rolling_std
        return z_score
    
    def generate_trading_signals(self, z_score: pd.Series) -> pd.Series:
        """
        Generate trading signals based on z-score thresholds
        
        Args:
            z_score: Z-score series
            
        Returns:
            Trading signals: 1 for long spread, -1 for short spread, 0 for no position
        """
        signals = pd.Series(index=z_score.index, data=0)
        
        # Entry signals
        signals[z_score > self.z_entry_threshold] = -1  # Short spread
        signals[z_score < -self.z_entry_threshold] = 1    # Long spread
        
        # Exit signals
        signals[abs(z_score) < self.z_exit_threshold] = 0
        
        return signals
    
    def backtest_pairs_strategy(self, stock1: str, stock2: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest pairs trading strategy for a given pair
        
        Args:
            stock1: Ticker symbol for first stock
            stock2: Ticker symbol for second stock
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Download data
        data1 = yf.download(stock1, start=start_date, end=end_date)
        data2 = yf.download(stock2, start=start_date, end=end_date)
        
        # Align data
        prices1 = data1['Close']
        prices2 = data2['Close']
        
        # Calculate spread and z-score
        spread = self.calculate_spread(prices1, prices2)
        z_score = self.calculate_z_score(spread)
        
        # Generate signals
        signals = self.generate_trading_signals(z_score)
        
        # Calculate returns
        returns1 = prices1.pct_change()
        returns2 = prices2.pct_change()
        
        # Portfolio returns (assuming equal dollar amounts)
        portfolio_returns = signals.shift(1) * (returns1 - returns2)
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
        
        # Win rate
        winning_trades = (portfolio_returns > 0).sum()
        total_trades = (signals != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'signals': signals,
            'z_score': z_score,
            'spread': spread
        }
    
    def find_best_pairs(self, universe: List[str], start_date: str, end_date: str, top_n: int = 5) -> List[Dict]:
        """
        Find the best performing pairs from a universe of stocks
        
        Args:
            universe: List of stock tickers
            start_date: Start date for analysis
            end_date: End date for analysis
            top_n: Number of top pairs to return
            
        Returns:
            List of dictionaries with pair performance metrics
        """
        # Download data for all stocks
        prices = pd.DataFrame()
        for stock in universe:
            try:
                data = yf.download(stock, start=start_date, end=end_date)
                prices[stock] = data['Close']
            except:
                continue
        
        # Find cointegrated pairs
        coint_pairs = self.find_cointegrated_pairs(prices)
        
        # Backtest each pair
        results = []
        for stock1, stock2, p_value in coint_pairs[:top_n*3]:  # Test more pairs
            try:
                backtest_result = self.backtest_pairs_strategy(stock1, stock2, start_date, end_date)
                backtest_result['pair'] = f"{stock1}-{stock2}"
                backtest_result['p_value'] = p_value
                results.append(backtest_result)
            except:
                continue
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        return results[:top_n]

# Example usage
if __name__ == "__main__":
    # Initialize pairs trading strategy
    pairs_trader = PairsTradingStrategy()
    
    # Define universe of stocks
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    # Find best pairs
    best_pairs = pairs_trader.find_best_pairs(
        universe=universe,
        start_date='2023-01-01',
        end_date='2024-01-01',
        top_n=3
    )
    
    # Print results
    for pair_result in best_pairs:
        print(f"Pair: {pair_result['pair']}")
        print(f"Sharpe Ratio: {pair_result['sharpe_ratio']:.3f}")
        print(f"Total Return: {pair_result['total_return']:.3f}")
        print(f"Win Rate: {pair_result['win_rate']:.2%}")
        print("-" * 40)
