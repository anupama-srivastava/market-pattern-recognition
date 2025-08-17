import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MarketDataset(Dataset):
    """PyTorch Dataset for market data sequences"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class AdvancedFeatureEngineer:
    """Enhanced feature engineering with advanced indicators"""
    
    def __init__(self):
        self.scalers = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Advanced Moving Averages
        df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        df['WMA_20'] = ta.trend.WMAIndicator(df['Close'], window=20).wma()
        df['HMA_20'] = self._hma(df['Close'], 20)
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['High'], 
            low=df['Low'],
            window1=9,
            window2=26,
            window3=52
        )
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Advanced RSI variants
        df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['Close'], window=21).rsi()
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['Close']).stochrsi()
        
        # Bollinger Bands with different deviations
        for dev in [2, 2.5, 3]:
            indicator = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=dev)
            df[f'bb_upper_{dev}'] = indicator.bollinger_hband()
            df[f'bb_lower_{dev}'] = indicator.bollinger_lband()
            df[f'bb_width_{dev}'] = indicator.bollinger_wband()
        
        # ATR and volatility measures
        df['atr_14'] = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).average_true_range()
        
        # Volume indicators
        df['volume_sma_20'] = ta.volume.VolumeSMAIndicator(df['Volume'], window=20).volume_sma()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['on_balance_volume'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], 
            volume=df['Volume']
        ).on_balance_volume()
        
        # Price momentum
        df['momentum_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).williams_r()
        
        # Support and resistance levels
        df['support_20'] = df['Low'].rolling(window=20).min()
        df['resistance_20'] = df['High'].rolling(window=20).max()
        
        # Fibonacci retracements
        df = self._add_fibonacci_levels(df)
        
        # Market regime indicators
        df['regime'] = self._detect_market_regime(df)
        
        return df
    
    def _hma(self, series: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average calculation"""
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        
        wma_half = ta.trend.WMAIndicator(series, window=half_length).wma()
        wma_full = ta.trend.WMAIndicator(series, window=period).wma()
        
        raw_hma = 2 * wma_half - wma_full
        hma = ta.trend.WMAIndicator(raw_hma, window=sqrt_length).wma()
        
        return hma
    
    def _add_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement levels"""
        lookback = 50
        
        high_50 = df['High'].rolling(window=lookback).max()
        low_50 = df['Low'].rolling(window=lookback).min()
        range_50 = high_50 - low_50
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            df[f'fib_{int(level*1000)}'] = low_50 + (range_50 * level)
        
        return df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime using volatility and trend"""
        volatility = df['returns'].rolling(window=20).std()
        trend = ta.trend.MACD(df['Close']).macd_diff()
        
        regime = pd.Series(index=df.index, dtype='int64')
        
        # Bull market: low volatility + positive trend
        regime[(volatility < volatility.quantile(0.3)) & (trend > 0)] = 0
        
        # Bear market: high volatility + negative trend
        regime[(volatility > volatility.quantile(0.7)) & (trend < 0)] = 1
        
        # Sideways market: everything else
        regime[regime.isna()] = 2
        
        return regime
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60, 
                        prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        # Select features for modeling
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure we have numeric data
        df_features = df[feature_cols].select_dtypes(include=[np.number]).dropna()
        
        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(df_features)
        self.scalers['features'] = scaler
        
        # Create target variable (next 5-day return)
        returns = df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        target = (returns > 0.02).astype(int)  # Binary classification
        
        # Align features and targets
        features_clean = features_scaled[:len(target)]
        target_clean = target.dropna()
        features_clean = features_clean[:len(target_clean)]
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(features_clean)):
            sequences.append(features_clean[i-sequence_length:i])
            targets.append(target_clean.iloc[i])
        
        return np.array(sequences), np.array(targets)

class LSTMMarketPredictor(nn.Module):
    """Advanced LSTM architecture for market prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.3, 
                 bidirectional: bool = True):
        super(LSTMMarketPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Use the last output with attention
        last_output = attn_output[-1, :, :]
        
        # Final classification
        output = self.fc_layers(last_output)
        
        return output

class GRUMarketPredictor(nn.Module):
    """GRU-based architecture as alternative to LSTM"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.3):
        super(GRUMarketPredictor, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        output = self.fc_layers(gru_out[:, -1, :])
        return output

class MarketModelTrainer:
    """Training pipeline for LSTM/GRU models"""
    
    def __init__(self, model_type: str = 'lstm', device: str = None):
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_engineer = AdvancedFeatureEngineer()
        
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 60,
                    batch_size: int = 32, test_size: float = 0.2) -> Dict:
        """Prepare data for training"""
        
        # Add advanced features
        df_features = self.feature_engineer.add_technical_indicators(df)
        
        # Create sequences
        X, y = self.feature_engineer.create_sequences(
            df_features, sequence_length=sequence_length
        )
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = MarketDataset(X_train, y_train)
        test_dataset = MarketDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'input_size': X.shape[2]
        }
    
    def build_model(self, input_size: int) -> nn.Module:
        """Build the specified model architecture"""
        
        if self.model_type.lower() == 'lstm':
            model = LSTMMarketPredictor(input_size=input_size)
        elif self.model_type.lower() == 'gru':
            model = GRUMarketPredictor(input_size=input_size)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        
        return model.to(self.device)
    
    def train_model(self, data_dict: Dict, epochs: int = 100, 
                   learning_rate: float = 0.001) -> Dict:
        """Train the model with advanced techniques"""
        
        model = self.build_model(data_dict['input_size'])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in data_dict['train_loader']:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in data_dict['test_loader']:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(data_dict['train_loader'])
            avg_val_loss = val_loss / len(data_dict['test_loader'])
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        self.model = model
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'feature_engineer': self.feature_engineer
        }
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(sequences_tensor)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_engineer': self.feature_engineer,
            'model_type': self.model_type
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.feature_engineer = checkpoint['feature_engineer']
        
        # Rebuild model architecture
        self.model = self.build_model(checkpoint['input_size'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download data
    ticker = "AAPL"
    data = yf.download(ticker, period="2y", interval="1d")
    
    # Initialize trainer
    trainer = MarketModelTrainer(model_type='lstm')
    
    # Prepare data
    data_dict = trainer.prepare_data(data, sequence_length=60)
    
    # Train model
    results = trainer.train_model(data_dict, epochs=50)
    
    # Save model
    trainer.save_model(f'models/lstm_{ticker}_model.pth')
    
    print("✅ LSTM model training completed!")
