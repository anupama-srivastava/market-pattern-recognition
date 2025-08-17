"""
Sentiment Analysis Module for Market Pattern Recognition
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

class NewsSentimentAnalyzer:
    """Advanced news sentiment analysis using transformer models"""
    
    def __init__(self, model_name: "microsoft/DialoGPT-medium"):
        """
        Initialize sentiment analyzer with transformer model
        
        Args:
            model_name: HuggingFace model for sentiment analysis
        """
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_news_sentiment(self, news_text: str) -> Dict[str, float]:
        """
        Analyze sentiment of news text using both transformer and VADER
        
        Args:
            news_text: Raw news text
            
        Returns:
            Dictionary with sentiment scores
        """
        # Transformer-based sentiment
        transformer_result = self.sentiment_pipeline(news_text[:512])[0]
        transformer_score = 1.0 if transformer_result['label'] == 'POSITIVE' else -1.0
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(news_text)
        
        # Combined sentiment score
        combined_score = (
            transformer_score * 0.7 +  # Weight transformer more heavily
            vader_scores['compound'] * 0.3
        )
        
        return {
            'transformer_sentiment': transformer_result['label'],
            'transformer_score': transformer_result['score'],
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'combined_score': combined_score
        }
    
    def get_news_sentiment_for_symbol(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get news sentiment for a specific stock symbol
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            DataFrame with sentiment scores and dates
        """
        # This would integrate with actual news APIs
        # For now, we'll create synthetic sentiment data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Generate synthetic sentiment data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
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

class SocialMediaSentimentAnalyzer:
    """Social media sentiment analysis from Twitter and Reddit"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_twitter_sentiment(self, tweets: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment from Twitter data
        
        Args:
            tweets: List of tweet texts
            
        Returns:
            Aggregated sentiment scores
        """
        sentiments = []
        
        for tweet in tweets:
            # Clean tweet text
            cleaned_tweet = self._clean_tweet_text(tweet)
            
            # Get sentiment
            scores = self.vader_analyzer.polarity_scores(cleaned_tweet)
            sentiments.append(scores['compound'])
        
        return {
            'mean_sentiment': np.mean(sentiments),
            'std_sentiment': np.std(sentiments),
            'positive_ratio': np.sum(np.array(sentiments) > 0.1) / len(sentiments),
            'negative_ratio': np.sum(np.array(sentiments) < -0.1) / len(sentiments),
            'neutral_ratio': np.sum(np.abs(np.array(sentiments)) <= 0.1) / len(sentiments)
        }
    
    def analyze_reddit_sentiment(self, posts: List[str], subreddit: str = "wallstreetbets") -> Dict[str, float]:
        """
        Analyze sentiment from Reddit posts
        
        Args:
            posts: List of Reddit post texts
            subreddit: Subreddit name
            
        Returns:
            Aggregated sentiment scores
        """
        sentiments = []
        
        for post in posts:
            # Clean post text
            cleaned_post = self._clean_reddit_text(post)
            
            # Get sentiment
            scores = self.vader_analyzer.polarity_scores(cleaned_post)
            sentiments.append(scores['compound'])
        
        return {
            'mean_sentiment': np.mean(sentiments),
            'std_sentiment': np.std(sentiments),
            'bullish_ratio': np.sum(np.array(sentiments) > 0.2) / len(sentiments),
            'bearish_ratio': np.sum(np.array(sentiments) < -0.2) / len(sentiments),
            'neutral_ratio': np.sum(np.abs(np.array(sentiments)) <= 0.2) / len(sentiments)
        }
    
    def _clean_tweet_text(self, text: str) -> str:
        """Clean Twitter text for sentiment analysis"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        text = text.lower().strip()
        return text
    
    def _clean_reddit_text(self, text: str) -> str:
        """Clean Reddit text for sentiment analysis"""
        # Remove URLs and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        text = text.lower().strip()
        return text

class EconomicSentimentAnalyzer:
    """Economic indicators and sentiment analysis"""
    
    def __init__(self):
        self.economic_indicators = [
            'GDP', 'CPI', 'FED_RATE', 'UNEMPLOYMENT', 'CONSUMER_CONFIDENCE'
        ]
    
    def get_economic_calendar_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get economic calendar sentiment
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with economic sentiment scores
        """
        # This would integrate with actual economic calendar APIs
        # For now, create synthetic data
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        economic_data = []
        for date in dates:
            # Simulate economic sentiment
            sentiment_score = np.random.normal(0, 0.2)
            
            economic_data.append({
                'date': date,
                'gdp_sentiment': np.random.normal(0, 0.1),
                'cpi_sentiment': np.random.normal(0, 0.15),
                'fed_rate_sentiment': np.random.normal(0, 0.2),
                'unemployment_sentiment': np.random.normal(0, 0.1),
                'consumer_confidence_sentiment': np.random.normal(0, 0.12),
                'overall_economic_sentiment': sentiment_score
            })
        
        return pd.DataFrame(economic_data)

class SentimentFeatureEngineer:
    """Integrate sentiment features into existing feature pipeline"""
    
    def __init__(self, news_analyzer: NewsSentimentAnalyzer, 
                 social_analyzer: SocialMediaSentimentAnalyzer,
                 economic_analyzer: EconomicSentimentAnalyzer):
        self.news_analyzer = news_analyzer
        self.social_analyzer = social_analyzer
        self.economic_analyzer = economic_analyzer
    
    def create_sentiment_features(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create sentiment-based features for a given symbol
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with sentiment features
        """
        # Get news sentiment
        news_sentiment = self.news_analyzer.get_news_sentiment_for_symbol(symbol, 30)
        
        # Get economic sentiment
        economic_sentiment = self.economic_analyzer.get_economic_calendar_sentiment(start_date, end_date)
        
        # Merge and align data
        sentiment_features = pd.merge(
            news_sentiment,
            economic_sentiment,
            on='date',
            how='inner'
        )
        
        # Create rolling sentiment features
        sentiment_features['sentiment_ma_7'] = sentiment_features['sentiment_score'].rolling(7).mean()
        sentiment_features['sentiment_ma_14'] = sentiment_features['sentiment_score'].rolling(14).mean()
        sentiment_features['sentiment_volatility'] = sentiment_features['sentiment_score'].rolling(7).std()
        
        # Create sentiment momentum
        sentiment_features['sentiment_momentum'] = sentiment_features['sentiment_score'].diff()
        
        # Create sentiment extremes
        sentiment_features['sentiment_extreme'] = (
            (sentiment_features['sentiment_score'] > sentiment_features['sentiment_score'].quantile(0.9)) |
            (sentiment_features['sentiment_score'] < sentiment_features['sentiment_score'].quantile(0.1))
        ).astype(int)
        
        return sentiment_features

# Example usage
if __name__ == "__main__":
    # Initialize analyzers
    news_analyzer = NewsSentimentAnalyzer("microsoft/DialoGPT-medium")
    social_analyzer = SocialMediaSentimentAnalyzer()
    economic_analyzer = EconomicSentimentAnalyzer()
    
    # Create sentiment features
    sentiment_engineer = SentimentFeatureEngineer(
        news_analyzer, social_analyzer, economic_analyzer
    )
    
    # Example: Get sentiment features for AAPL
    features = sentiment_engineer.create_sentiment_features(
        "AAPL", "2023-01-01", "2023-12-31"
    )
    
    print("✅ Sentiment features created successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
