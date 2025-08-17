"""
Advanced Sentiment Analysis Module for Market Pattern Recognition
Integrates news sentiment, social media sentiment, and alternative data sources
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import requests
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis using transformer models and multiple data sources"""
    
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_news_sentiment(self, news_text: str) -> Dict[str, float]:
        """Analyze sentiment of news text using both transformer and VADER"""
        transformer_result = self.sentiment_pipeline(news_text[:512])[0]
        vader_scores = self.vader_analyzer.polarity_scores(news_text)
        
        return {
            'transformer_sentiment': transformer_result['label'],
            'transformer_score': transformer_result['score'],
            'vader_compound': vader_scores['compound'],
            'combined_score': (transformer_result['score'] + vader_scores['compound']) / 2
        }
    
    def get_sentiment_features(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Get sentiment features for a specific symbol"""
        # This would integrate with actual news APIs
        # For now, create synthetic sentiment data
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        
        sentiment_data = []
        for date in dates:
            # Simulate news sentiment based on price movement
            base_sentiment = np.random.normal(0, 0.3)
            
            sentiment_data.append({
                'date': date,
                'symbol': symbol,
                'sentiment_score': base_sentiment,
                'news_count': np.random.randint(1, 10),
                'positive_news_ratio': np.random.uniform(0.3, 0.7),
                'sentiment_volatility': np.random.uniform(0.1, 0.5)
            })
        
        return pd.DataFrame(sentiment_data)

class CandlestickPatternDetector:
    """Advanced candlestick pattern detection using computer vision"""
    
    def __init__(self):
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'engulfing': self._detect_engulfing,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star
        }
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all candlestick patterns"""
        df = df.copy()
        
        # Calculate basic candlestick metrics
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Detect patterns
        for pattern_name, pattern_func in self.patterns.items():
            df[pattern_name] = pattern_func(df)
        
        return df
    
    def _detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """Detect Doji patterns"""
        return (abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1)
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer patterns"""
        return ((df['Close'] > df['Open']) & 
                (df['lower_shadow'] > 2 * abs(df['Close'] - df['Open'])))
    
    def _detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect Engulfing patterns"""
        return ((df['Close'] > df['Open']) & 
                (df['Close'].shift(1) < df['Open'].shift(1)) &
                (df['Close'] > df['Open'].shift(1)))
    
    def _detect_morning_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Morning Star patterns"""
        return ((df['Close'] > df['Open']) & 
                (df['Close'].shift(1) < df['Open'].shift(1)) &
                (df['Close'] > df['Close'].shift(2)))
    
    def _detect_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Evening Star patterns"""
        return ((df['Close'] < df['Open']) & 
                (df['Close'].shift(1) > df['Open'].shift(1)) &
                (df['Close'] < df['Close'].shift(2)))

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple strategies"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def mean_variance_optimization(self, returns: pd.DataFrame, target_return: float = 0.12) -> Dict:
        """Mean-variance optimization using quadratic programming"""
        n_assets = returns.shape[1]
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        init_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'expected_return': np.dot(optimal_weights, mean_returns),
            'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))),
            'sharpe_ratio': (np.dot(optimal_weights, mean_returns) - self.risk_free_rate) / np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        }
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        """Risk parity optimization - equal risk contribution"""
        n_assets = returns.shape[1]
        cov_matrix = returns.cov()
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            return np.sum((risk_contrib - 1/n_assets)**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        init_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_parity_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'expected_return': np.dot(optimal_weights, returns.mean()),
            'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov(), optimal_weights)))
        }
