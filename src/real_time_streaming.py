"""
Real-time Streaming with Kafka Module

This module implements real-time data streaming with Kafka for market data processing.
"""

import json
import time
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import yfinance as yf
import pandas as pd
import threading
import logging

logging.basicConfig(level=logging.INFO)

class KafkaStreamingService:
    """Real-time streaming service with Kafka"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
        
    def create_producer(self):
        """Create Kafka producer"""
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: json.dumps(k).encode('utf-8')
        )
        
    def create_consumer(self, topic: str, group_id: str = 'market-data-group'):
        """Create Kafka consumer"""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: json.loads(k.decode('utf-8'))
        )
        
    def produce_market_data(self, symbol: str, topic: str = 'market-data'):
        """Produce real-time market data to Kafka"""
        ticker = yf.Ticker(symbol)
        
        while True:
            try:
                # Get real-time data
                data = ticker.history(period='1d', interval='1m')
                
                if not data.empty:
                    latest = data.iloc[-1]
                    message = {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'rsi': 50,  # Placeholder
                        'macd': 0   # Placeholder
                    }
                    
                    self.producer.send(topic, message)
                    logging.info(f"Produced data for {symbol}: {message}")
                    
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logging.error(f"Error producing data: {e}")
                
    def consume_market_data(self, topic: str = 'market-data'):
        """Consume real-time market data from Kafka"""
        for message in self.consumer:
            try:
                data = message.value
                logging.info(f"Received data: {data}")
                
                # Process the data
                self._process_data(data)
                
            except Exception as e:
                logging.error(f"Error processing data: {e}")
                
    def _process_data(self, data: dict):
        """Process received data"""
        # Add processing logic here
        logging.info(f"Processing: {data}")
        
    def start_streaming(self, symbols: list, topic: str = 'market-data'):
        """Start streaming data for given symbols"""
        self.create_producer()
        self.create_consumer(topic)
        
        # Start producer in separate thread
        producer_thread = threading.Thread(
            target=self.produce_market_data,
            args=(symbols[0], topic)
        )
        producer_thread.start()
        
        # Start consumer in separate thread
        consumer_thread = threading.Thread(
            target=self.consume_market_data,
            args=(topic,)
        )
        consumer_thread.start()

class KafkaDataProcessor:
    """Data processing with Kafka"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        
    def process_stream(self, topic: str):
        """Process streaming data"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for message in consumer:
            try:
                data = message.value
                # Process data
                logging.info(f"Processing: {data}")
                
            except Exception as e:
                logging.error(f"Error processing data: {e}")

# Example usage
if __name__ == "__main__":
    streaming_service = KafkaStreamingService()
    streaming_service.start_streaming(['AAPL', 'GOOGL', 'MSFT'])
