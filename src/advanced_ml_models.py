"""
Advanced Machine Learning Models Module

This module implements transformer-based models and reinforcement learning for market prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf

class MarketTransformer(nn.Module):
    """Transformer-based model for market prediction"""
    
    def __init__(self, input_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super(MarketTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes: bearish, neutral, bullish
        )
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Apply transformer
        output = self.transformer(x)
        
        # Use the last output for classification
        output = output[:, -1, :]
        
        # Final classification
        return self.classifier(output)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        return self._get_observation()
        
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            return np.zeros(10)
            
        row = self.data.iloc[self.current_step]
        return np.array([
            row['Close'],
            row['Volume'],
            row.get('RSI', 50),
            row.get('MACD', 0),
            row.get('BB_upper', 0),
            row.get('BB_lower', 0),
            row.get('SMA_20', 0),
            row.get('SMA_50', 0),
            row.get('ATR', 0),
            self.position
        ], dtype=np.float32)
        
    def step(self, action):
        """Execute one step in the environment"""
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
                self.trades.append(('buy', current_price, self.current_step))
                
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                self.trades.append(('sell', current_price, self.current_step))
        
        # Calculate reward
        reward = 0
        if self.current_step > 0:
            prev_price = self.data.iloc[self.current_step - 1]['Close']
            if self.position > 0:
                reward = (current_price - prev_price) / prev_price * 100
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate total value
        total_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        
        return self._get_observation(), reward, done, {
            'balance': self.balance,
            'position': self.position,
            'total_value': total_value,
            'trades': self.trades
        }

class RLTradingAgent:
    """Reinforcement Learning trading agent"""
    
    def __init__(self, model_type: str = 'PPO'):
        self.model_type = model_type
        self.model = None
        
    def train(self, env: TradingEnvironment, total_timesteps: int = 10000):
        """Train the RL agent"""
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize model
        if self.model_type == 'PPO':
            self.model = PPO('MlpPolicy', vec_env, verbose=1)
        elif self.model_type == 'A2C':
            self.model = A2C('MlpPolicy', vec_env, verbose=1)
        elif self.model_type == 'DQN':
            self.model = DQN('MlpPolicy', vec_env, verbose=1)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
            
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
    def predict(self, observation: np.ndarray):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.predict(observation)

class AdvancedModelTrainer:
    """Trainer for advanced ML models"""
    
    def __init__(self):
        self.transformer_model = None
        self.rl_agent = None
        
    def prepare_transformer_data(self, data: pd.DataFrame, 
                                sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for transformer training"""
        
        # Add technical indicators
        data['RSI'] = data['Close'].pct_change().rolling(14).mean()
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        
        # Create sequences
        features = ['Close', 'Volume', 'RSI', 'MACD']
        X = data[features].values
        
        # Create target (next day direction)
        y = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            targets.append(y.iloc[i])
            
        return np.array(sequences), np.array(targets)
        
    def train_transformer(self, X: np.ndarray, y: np.ndarray, 
                         epochs: int = 100, batch_size: int = 32):
        """Train transformer model"""
        
        # Convert to 3 classes
        y_3class = np.zeros((len(y), 3))
        for i, val in enumerate(y):
            if val == 0:
                y_3class[i] = [1, 0, 0]  # bearish
            else:
                y_3class[i] = [0, 0, 1]  # bullish
                
        # Initialize model
        input_size = X.shape[2]
        self.transformer_model = MarketTransformer(input_size=input_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            # Training code here
            pass
            
    def train_rl_agent(self, data: pd.DataFrame, model_type: str = 'PPO'):
        """Train reinforcement learning agent"""
        
        env = TradingEnvironment(data)
        self.rl_agent = RLTradingAgent(model_type)
        self.rl_agent.train(env)

# Example usage
if __name__ == "__main__":
    # Download sample data
    data = yf.download('AAPL', period='1y', interval='1d')
    
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Train RL agent
    trainer.train_rl_agent(data, model_type='PPO')
    
    print("✅ Advanced ML models training completed!")
