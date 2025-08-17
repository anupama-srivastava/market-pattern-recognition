"""
Feature Engineering Service
Provides real-time feature extraction and technical indicators
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from pydantic import BaseModel
import yfinance as yf
from src.feature_engineering import TechnicalIndicators
from src.advanced_features import AdvancedFeatureEngine
from src.advanced_patterns import AdvancedPatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Feature Engineering Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize feature engines
technical_indicators = TechnicalIndicators()
advanced_features = AdvancedFeatureEngine()
pattern_detector = AdvancedPatternDetector()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        os.getenv("POSTGRES_URL", "postgresql://market_user:market_pass@localhost:5432/market_patterns")
    )

# Redis connection
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

# Pydantic models
class FeatureRequest(BaseModel):
    symbol: str
    period: str = "1y"
    indicators: List[str] = ["sma", "ema", "rsi", "macd"]

class PatternRequest(BaseModel):
    symbol: str
    pattern_type: str = "candlestick"

class PredictionRequest(BaseModel):
    symbol: str
    features: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feature_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        sma_20 FLOAT,
                        sma_50 FLOAT,
                        ema_12 FLOAT,
                        ema_26 FLOAT,
                        rsi FLOAT,
                        macd FLOAT,
                        macd_signal FLOAT,
                        bollinger_upper FLOAT,
                        bollinger_lower FLOAT,
                        volume_sma FLOAT,
                        atr FLOAT,
                        stochastic_k FLOAT,
                        stochastic_d FLOAT,
                        williams_r FLOAT,
                        adx FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        pattern_type VARCHAR(50),
                        pattern_name VARCHAR(100),
                        confidence FLOAT,
                        price FLOAT,
                        volume BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_data_symbol_timestamp 
                    ON feature_data(symbol, timestamp)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pattern_data_symbol_timestamp 
                    ON pattern_data(symbol, timestamp)
                """)
            conn.commit()
        logger.info("Feature engineering database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Feature Engineering Service is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check PostgreSQL connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/features/calculate")
async def calculate_features(request: FeatureRequest):
    """Calculate technical indicators for a symbol"""
    try:
        # Get historical data
        ticker = yf.Ticker(request.symbol)
        data = ticker.history(period=request.period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Calculate features
        features = {}
        
        if "sma" in request.indicators:
            features.update({
                "sma_20": technical_indicators.sma(data["Close"], 20).iloc[-1],
                "sma_50": technical_indicators.sma(data["Close"], 50).iloc[-1]
            })
        
        if "ema" in request.indicators:
            features.update({
                "ema_12": technical_indicators.ema(data["Close"], 12).iloc[-1],
                "ema_26": technical_indicators.ema(data["Close"], 26).iloc[-1]
            })
        
        if "rsi" in request.indicators:
            features["rsi"] = technical_indicators.rsi(data["Close"]).iloc[-1]
        
        if "macd" in request.indicators:
            macd_line, signal_line = technical_indicators.macd(data["Close"])
            features.update({
                "macd": macd_line.iloc[-1],
                "macd_signal": signal_line.iloc[-1]
            })
        
        if "bollinger" in request.indicators:
            upper, lower = technical_indicators.bollinger_bands(data["Close"])
            features.update({
                "bollinger_upper": upper.iloc[-1],
                "bollinger_lower": lower.iloc[-1]
            })
        
        if "volume" in request.indicators:
            features["volume_sma"] = technical_indicators.sma(data["Volume"], 20).iloc[-1]
        
        if "atr" in request.indicators:
            features["atr"] = technical_indicators.atr(data).iloc[-1]
        
        if "stochastic" in request.indicators:
            k, d = technical_indicators.stochastic(data)
            features.update({
                "stochastic_k": k.iloc[-1],
                "stochastic_d": d.iloc[-1]
            })
        
        if "williams" in request.indicators:
            features["williams_r"] = technical_indicators.williams_r(data).iloc[-1]
        
        if "adx" in request.indicators:
            features["adx"] = technical_indicators.adx(data).iloc[-1]
        
        # Store in PostgreSQL
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feature_data (symbol, timestamp, sma_20, sma_50, ema_12, ema_26, rsi, macd, macd_signal, bollinger_upper, bollinger_lower, volume_sma, atr, stochastic_k, stochastic_d, williams_r, adx)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    request.symbol,
                    datetime.now().isoformat(),
                    features.get("sma_20"),
                    features.get("sma_50"),
                    features.get("ema_12"),
                    features.get("ema_26"),
                    features.get("rsi"),
                    features.get("macd"),
                    features.get("macd_signal"),
                    features.get("bollinger_upper"),
                    features.get("bollinger_lower"),
                    features.get("volume_sma"),
                    features.get("atr"),
                    features.get("stochastic_k"),
                    features.get("stochastic_d"),
                    features.get("williams_r"),
                    features.get("adx")
                ))
            conn.commit()
        
        # Cache in Redis
        redis_key = f"features:{request.symbol}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
        redis_client.setex(redis_key, 3600, json.dumps(features))
        
        return {
            "symbol": request.symbol,
            "features": features,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/detect")
async def detect_patterns(request: PatternRequest):
    """Detect market patterns for a symbol"""
    try:
        # Get historical data
        ticker = yf.Ticker(request.symbol)
        data = ticker.history(period="1y")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Detect patterns
        patterns = []
        
        if request.pattern_type == "candlestick":
            candlestick_patterns = pattern_detector.detect_candlestick_patterns(data)
            for pattern_name, confidence, price, volume in candlestick_patterns:
                pattern_record = {
                    "symbol": request.symbol,
                    "timestamp": datetime.now().isoformat(),
                    "pattern_type": "candlestick",
                    "pattern_name": pattern_name,
                    "confidence": confidence,
                    "price": price,
                    "volume": volume
                }
                patterns.append(pattern_record)
        
        # Store in PostgreSQL
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO pattern_data (symbol, timestamp, pattern_type, pattern_name, confidence, price, volume)
                    VALUES (%(symbol)s, %(timestamp)s, %(pattern_type)s, %(pattern_name)s, %(confidence)s, %(price)s, %(volume)s)
                """, patterns)
            conn.commit()
        
        # Cache in Redis
        for pattern in patterns:
            redis_key = f"pattern:{request.symbol}:{pattern['pattern_name']}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
            redis_client.setex(redis_key, 3600, json.dumps(pattern))
        
        return {
            "symbol": request.symbol,
            "patterns": patterns,
            "count": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/{symbol}")
async def get_features(symbol: str, limit: int = 100):
    """Get calculated features for a symbol"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM feature_data 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (symbol, limit))
                data = cur.fetchall()
        
        return {"symbol": symbol, "features": data, "count": len(data)}
        
    except Exception as e:
        logger.error(f"Error retrieving features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patterns/{symbol}")
async def get_patterns(symbol: str, limit: int = 100):
    """Get detected patterns for a symbol"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM pattern_data 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (symbol, limit))
                data = cur.fetchall()
        
        return {"symbol": symbol, "patterns": data, "count": len(data)}
        
    except Exception as e:
        logger.error(f"Error retrieving patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/{key}")
async def get_cache_value(key: str):
    """Get value from Redis cache"""
    try:
        value = redis_client.get(key)
        if value:
            return {"key": key, "value": json.loads(value)}
        return {"key": key, "value": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
