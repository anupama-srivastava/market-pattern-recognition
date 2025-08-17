"""
Real-time data pipeline for market pattern recognition
Phase 1: Foundation implementation
"""

import asyncio
import websockets
import json
import redis
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import threading
import time
from collections import deque

class RealTimeDataProcessor:
    """Real-time market data processing and feature extraction"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.data_buffer = deque(maxlen=1000)  # Keep last 1000 data points
        self.feature_cache = {}
        
    async def connect_websocket(self, symbol: str):
        """Connect to real-time market data WebSocket"""
        # Example WebSocket URL (replace with actual provider)
        uri = f"wss://stream.data-provider.com/v1/market/{symbol}"
        
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "type": "subscribe",
                "symbol": symbol,
                "fields": ["price", "volume", "timestamp"]
            }))
            
            async for message in websocket:
                data = json.loads(message)
                await self.process_realtime_data(data)
    
    async def process_realtime_data(self, data: Dict):
        """Process incoming real-time market data"""
        timestamp = datetime.fromisoformat(data['timestamp'])
        price = float(data['price'])
        volume = int(data['volume'])
        
        # Store in Redis with TTL
        key = f"market:{data['symbol']}:{timestamp.strftime('%Y%m%d%H%M%S')}"
        self.redis_client.setex(key, 3600, json.dumps({
            'price': price,
            'volume': volume,
            'timestamp': str(timestamp)
        }))
        
        # Update buffer
        self.data_buffer.append({
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        })
        
        # Calculate real-time features
        features = self.calculate_realtime_features()
        
        # Cache features for model inference
        self.feature_cache[data['symbol']] = features
        
    def calculate_realtime_features(self) -> Dict:
        """Calculate real-time technical indicators"""
        if len(self.data_buffer) < 20:
            return {}
        
        df = pd.DataFrame(list(self.data_buffer))
        df['returns'] = df['price'].pct_change()
        
        # Real-time indicators
        features = {
            'sma_20': df['price'].tail(20).mean(),
            'price_change_1m': (df['price'].iloc[-1] - df['price'].iloc[-60]) / df['price'].iloc[-60] if len(df) >= 60 else 0,
            'volatility': df['returns'].tail(20).std(),
            'volume_ratio': df['volume'].iloc[-1] / df['volume'].tail(20).mean()
        }
        
        return features
    
    def get_latest_features(self, symbol: str) -> Dict:
        """Get latest calculated features for a symbol"""
        return self.feature_cache.get(symbol, {})

class StreamingFeatureEngine:
    """Streaming feature engineering for real-time predictions"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.feature_windows = {}
        
    def update_features(self, symbol: str, new_data: Dict):
        """Update features with new data point"""
        if symbol not in self.feature_windows:
            self.feature_windows[symbol] = deque(maxlen=self.window_size)
        
        self.feature_windows[symbol].append(new_data)
        
        # Calculate streaming features
        features = self.calculate_streaming_features(symbol)
        return features
    
    def calculate_streaming_features(self, symbol: str) -> Dict:
        """Calculate features from streaming window"""
        if symbol not in self.feature_windows or len(self.feature_windows[symbol]) < 20:
            return {}
        
        window = list(self.feature_windows[symbol])
        prices = [d['price'] for d in window]
        volumes = [d['volume'] for d in window]
        
        # Streaming technical indicators
        features = {
            'current_price': prices[-1],
            'sma_20': np.mean(prices[-20:]),
            'ema_12': self._ema(prices, 12),
            'rsi': self._streaming_rsi(prices),
            'volume_sma': np.mean(volumes[-20:]),
            'price_momentum': (prices[-1] - prices[-10]) / prices[-10]
        }
        
        return features
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[-period:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _streaming_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI from streaming prices"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class DataPipelineManager:
    """Manage the complete real-time data pipeline"""
    
    def __init__(self):
        self.processor = RealTimeDataProcessor()
        self.feature_engine = StreamingFeatureEngine()
        self.running = False
        
    def start_pipeline(self, symbols: List[str]):
        """Start the real-time pipeline for given symbols"""
        self.running = True
        
        # Start WebSocket connections in separate threads
        for symbol in symbols:
            thread = threading.Thread(
                target=self._run_symbol_pipeline,
                args=(symbol,),
                daemon=True
            )
            thread.start()
    
    def _run_symbol_pipeline(self, symbol: str):
        """Run pipeline for a single symbol"""
        while self.running:
            try:
                # Simulate real-time data (replace with actual WebSocket)
                asyncio.run(self.processor.connect_websocket(symbol))
            except Exception as e:
                print(f"Error in {symbol} pipeline: {e}")
                time.sleep(5)  # Retry after 5 seconds
    
    def stop_pipeline(self):
        """Stop all pipeline processes"""
        self.running = False
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            'running': self.running,
            'symbols_processed': len(self.processor.feature_cache),
            'latest_data_points': len(self.processor.data_buffer)
        }

if __name__ == "__main__":
    # Example usage
    pipeline = DataPipelineManager()
    pipeline.start_pipeline(['AAPL', 'GOOGL', 'MSFT'])
    
    # Run for 60 seconds
    time.sleep(60)
    pipeline.stop_pipeline()
    
    print("Pipeline status:", pipeline.get_pipeline_status())
