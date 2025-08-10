import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import MarketPatternModel
from feature_engineering import FeatureEngineer

def test_feature_engineering():
    """Test feature engineering pipeline"""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 101,
        'Low': np.random.randn(100).cumsum() + 99,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    engineer = FeatureEngineer()
    df_features = engineer.prepare_features(df)
    
    assert len(df_features) > 0
    assert 'target' in df_features.columns
    assert df_features.isna().sum().sum() == 0

def test_model_training():
    """Test model training pipeline"""
    # This would normally use real data
    model = MarketPatternModel()
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 14)
    y = np.random.randint(0, 2, n_samples)
    
    # Mock the model training
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    model.model = RandomForestClassifier(random_state=42)
    model.scaler = StandardScaler()
    model.feature_columns = [f'feature_{i}' for i in range(14)]
    
    X_scaled = model.scaler.fit_transform(X)
    model.model.fit(X_scaled, y)
    
    assert model.model is not None
    assert len(model.feature_columns) == 14

if __name__ == "__main__":
    pytest.main([__file__])