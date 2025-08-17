"""
Data Ingestion Service
Provides real-time market data ingestion and storage
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import json
import logging
from typing import List, Dict, Optional
import yfinance as yf
import pandas as pd
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Data Ingestion Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
class SymbolRequest(BaseModel):
    symbol: str
    period: str = "1y"
    interval: str = "1d"

class WebSocketRequest(BaseModel):
    symbols: List[str]
    fields: List[str] = ["price", "volume", "timestamp"]

# Background task for real-time data collection
async def collect_realtime_data(symbols: List[str]):
    """Collect real-time market data for given symbols"""
    while True:
        try:
            for symbol in symbols:
                # Get real-time data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    latest = data.iloc[-1]
                    record = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "open": float(latest["Open"]),
                        "high": float(latest["High"]),
                        "low": float(latest["Low"]),
                        "close": float(latest["Close"]),
                        "volume": int(latest["Volume"])
                    }
                    
                    # Store in Redis
                    redis_key = f"market:{symbol}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    redis_client.setex(redis_key, 3600, json.dumps(record))
                    
                    # Store in PostgreSQL
                    with get_db_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                symbol,
                                record["timestamp"],
                                record["open"],
                                record["high"],
                                record["low"],
                                record["close"],
                                record["volume"]
                            ))
                        conn.commit()
                
                await asyncio.sleep(1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error collecting real-time data: {e}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        open FLOAT,
                        high FLOAT,
                        low FLOAT,
                        close FLOAT,
                        volume BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                    ON market_data(symbol, timestamp)
                """)
            conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Data Ingestion Service is running"}

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

@app.post("/ingest/historical")
async def ingest_historical_data(request: SymbolRequest):
    """Ingest historical market data"""
    try:
        ticker = yf.Ticker(request.symbol)
        data = ticker.history(period=request.period, interval=request.interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        records = []
        for date, row in data.iterrows():
            record = {
                "symbol": request.symbol,
                "timestamp": date.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            }
            records.append(record)
        
        # Store in PostgreSQL
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
                    VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                """, records)
            conn.commit()
        
        # Cache in Redis
        for record in records:
            redis_key = f"market:{request.symbol}:{record['timestamp']}"
            redis_client.setex(redis_key, 86400, json.dumps(record))
        
        return {
            "message": f"Successfully ingested {len(records)} records for {request.symbol}",
            "symbol": request.symbol,
            "records_count": len(records)
        }
        
    except Exception as e:
        logger.error(f"Error ingesting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/realtime/start")
async def start_realtime_ingestion(request: WebSocketRequest, background_tasks: BackgroundTasks):
    """Start real-time data ingestion"""
    background_tasks.add_task(collect_realtime_data, request.symbols)
    return {"message": "Real-time data ingestion started", "symbols": request.symbols}

@app.get("/data/{symbol}")
async def get_market_data(symbol: str, limit: int = 100):
    """Get market data for a symbol"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (symbol, limit))
                data = cur.fetchall()
        
        return {"symbol": symbol, "data": data, "count": len(data)}
        
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_available_symbols():
    """Get list of available symbols"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM market_data ORDER BY symbol")
                symbols = [row[0] for row in cur.fetchall()]
        
        return {"symbols": symbols}
        
    except Exception as e:
        logger.error(f"Error retrieving symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
