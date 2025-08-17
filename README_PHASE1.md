# Phase 1: Advanced Market Pattern Recognition - Deep Learning Foundation

## Overview
Phase 1 implements the deep learning foundation for advanced market pattern recognition, featuring LSTM/GRU networks with attention mechanisms, comprehensive feature engineering, and production-ready evaluation metrics.

## 🚀 Features Implemented

### 1. Deep Learning Architecture
- **LSTM with Attention**: Bidirectional LSTM with multi-head attention
- **GRU Alternative**: Gated Recurrent Unit architecture
- **Advanced Features**: 50+ technical indicators including Ichimoku Cloud, VWAP, Fibonacci levels
- **Market Regime Detection**: Bull/bear/sideways market classification

### 2. Real-time Data Pipeline
- **Redis Integration**: Low-latency feature serving
- **WebSocket Support**: Real-time market data streaming
- **Streaming Features**: Real-time technical indicator calculation
- **Microservices Ready**: Scalable architecture design

### 3. Advanced Evaluation
- **Walk-forward Analysis**: Realistic backtesting with rolling windows
- **Risk Metrics**: Sharpe ratio, max drawdown, Calmar ratio
- **Trading Performance**: Win rate, profit factor, returns analysis
- **Model Registry**: Version control and A/B testing

### 4. Production Features
- **Comprehensive Logging**: Detailed training and inference logs
- **Configuration Management**: JSON-based configuration
- **Model Persistence**: Save/load trained models
- **Error Handling**: Robust error handling and recovery

## 📁 Project Structure

```
market-pattern-recognition/
├── src/
│   ├── lstm_model.py          # LSTM/GRU implementation
│   ├── real_time_pipeline.py  # Real-time data processing
│   ├── advanced_evaluation.py # Evaluation metrics & walk-forward
│   ├── data_collection.py     # Data download utilities
│   └── feature_engineering.py # Feature engineering
├── models/
│   ├── phase1/               # Trained models
│   └── registry/             # Model registry
├── data/
│   ├── raw/                  # Raw market data
│   └── processed/            # Processed features
├── train_phase1.py           # Main training script
├── config_phase1.json        # Configuration file
└── requirements_phase1.txt   # Dependencies
```

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements_phase1.txt
```

### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ta; print('Technical analysis library ready')"
```

## 🎯 Quick Start

### 1. Basic Training
```bash
# Train with default configuration
python train_phase1.py

# Train specific symbols
python train_phase1.py --symbols AAPL GOOGL --model-type lstm --epochs 50

# Use custom configuration
python train_phase1.py --config config_phase1.json
```

### 2. Real-time Pipeline
```bash
# Start real-time data processing
python -c "
from src.real_time_pipeline import DataPipelineManager
manager = DataPipelineManager()
manager.start_pipeline(['AAPL', 'GOOGL'])
"
```

### 3. Model Evaluation
```bash
# Run walk-forward analysis
python -c "
from src.advanced_evaluation import WalkForwardAnalysis
from src.lstm_model import MarketModelTrainer
import yfinance as yf

data = yf.download('AAPL', period='2y')
trainer = MarketModelTrainer('lstm')
wfa = WalkForwardAnalysis(trainer, data)
results = wfa.run_walk_forward()
print(f'Average Sharpe: {results[\"avg_sharpe\"]:.2f}')
"
```

## 📊 Configuration

### Basic Configuration (`config_phase1.json`)
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
  "period": "2y",
  "model_type": "lstm",
  "sequence_length": 60,
  "prediction_horizon": 5,
  "epochs": 100,
  "walk_forward": {
    "enabled": true,
    "train_window": 252,
    "test_window": 63
  }
}
```

### Advanced Features
- **Technical Indicators**: 50+ indicators including Ichimoku, VWAP, Fibonacci
- **Market Regime Detection**: Automatic bull/bear/sideways classification
- **Risk Management**: Position sizing based on volatility
- **Feature Engineering**: Advanced momentum and volume indicators

## 🔍 Model Architecture

### LSTM with Attention
```python
class LSTMMarketPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### Feature Engineering Pipeline
- **Price Features**: Returns, log returns, volatility
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Ichimoku
- **Volume Analysis**: Volume ratios, OBV, VWAP
- **Market Microstructure**: Support/resistance levels

## 📈 Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: Annual return / max drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

### Walk-forward Analysis
- Rolling window backtesting
- Out-of-sample validation
- Regime-specific performance

## 🚀 Production Deployment

### 1. Model Training
```bash
# Full training pipeline
python train_phase1.py --config config_phase1.json

# Monitor training
tail -f training.log
```

### 2. Model Registry
```python
from src.advanced_evaluation import ModelRegistry
registry = ModelRegistry()
best_model = registry.get_best_model(metric='sharpe_ratio')
```

### 3. Real-time Inference
```python
from src.real_time_pipeline import DataPipelineManager
manager = DataPipelineManager()
features = manager.processor.get_latest_features('AAPL')
```

## 📊 Example Results

### Sample Training Output
```
Epoch 50/100: Train Loss = 0.2345, Val Loss = 0.2891
✅ AAPL model completed
- Accuracy: 0.742
- Sharpe Ratio: 1.89
- Max Drawdown: -12.3%
```

### Walk-forward Analysis
```
Walk-forward Summary:
- Total Periods: 8
- Average Sharpe: 1.67
- Win Rate: 75%
- Best Period: Q3 2023 (Sharpe: 2.34)
```

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Data Download Failures**: Check internet connection and yfinance limits
3. **Feature Calculation Errors**: Ensure sufficient data history

### Performance Optimization
- Use GPU acceleration: `CUDA_VISIBLE_DEVICES=0 python train_phase1.py`
- Reduce sequence length for faster training
- Use smaller symbol set for initial testing

## 📋 Next Steps (Phase 2)

- [ ] Sentiment analysis integration
- [ ] Advanced candlestick pattern recognition
- [ ] Multi-asset portfolio optimization
- [ ] Options flow analysis
- [ ] Production dashboard with Streamlit

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-indicator`
3. Add tests for new features
4. Submit pull request with performance metrics

## 📄 License

MIT License - See LICENSE file for details
