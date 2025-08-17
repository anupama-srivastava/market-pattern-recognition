"""
Model Serving Service
Provides ML model inference and predictions
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
import logging
import pickle
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Model Serving Service", version="1.0.0")

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

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    features: Dict[str, float]
    model_name: str = "market_pattern_model"

class BatchPredictionRequest(BaseModel):
    symbols: List[str]
    model_name: str = "market_pattern_model"

class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str

@app.on_event("startup")
async def startup_event():
    """Initialize model serving on startup"""
    try:
        # Set up MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        
        # Create predictions table
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        model_name VARCHAR(100) NOT NULL,
                        model_version VARCHAR(50),
                        prediction FLOAT,
                        probability FLOAT,
                        features JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp 
                    ON predictions(symbol, timestamp)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_model_name 
                    ON predictions(model_name)
                """)
            conn.commit()
        logger.info("Model serving database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Model Serving Service is running"}

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
        
        # Check MLflow connection
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.list_registered_models()
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        models = mlflow_client.list_registered_models()
        
        model_list = []
        for model in models:
            latest_version = mlflow_client.get_latest_versions(model.name, stages=["Production"])
            if latest_version:
                model_list.append({
                    "name": model.name,
                    "version": latest_version[0].version,
                    "stage": latest_version[0].current_stage
                })
        
        return {"models": model_list}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get model information"""
    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        model = mlflow_client.get_registered_model(model_name)
        
        versions = mlflow_client.get_latest_versions(model_name, stages=["Production", "Staging"])
        
        return {
            "name": model.name,
            "description": model.description,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "creation_timestamp": v.creation_timestamp
                }
                for v in versions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a prediction using the specified model"""
    try:
        # Load model from MLflow
        mlflow_client = mlflow.tracking.MlflowClient()
        model_uri = f"models:/{request.model_name}/Production"
        
        # Load model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Prepare features
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(features_df)[0].tolist()
        except:
            probability = None
        
        # Store prediction
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions (symbol, timestamp, model_name, model_version, prediction, probability, features)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    request.symbol,
                    datetime.now().isoformat(),
                    request.model_name,
                    mlflow_client.get_latest_versions(request.model_name, stages=["Production"])[0].version,
                    float(prediction),
                    json.dumps(probability) if probability else None,
                    json.dumps(request.features)
                ))
            conn.commit()
        
        # Cache in Redis
        redis_key = f"prediction:{request.symbol}:{request.model_name}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
        redis_client.setex(redis_key, 3600, json.dumps({
            "prediction": float(prediction),
            "probability": probability,
            "features": request.features
        }))
        
        return {
            "symbol": request.symbol,
            "model_name": request.model_name,
            "prediction": float(prediction),
            "probability": probability,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        # Load model from MLflow
        mlflow_client = mlflow.tracking.MlflowClient()
        model_uri = f"models:/{request.model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get features for all symbols
        features_list = []
        for symbol in request.symbols:
            # This would typically fetch features from feature store
            # For now, using placeholder features
            features = {"feature1": 1.0, "feature2": 2.0}  # Placeholder
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Make predictions
        predictions = model.predict(features_df)
        
        results = []
        for i, (symbol, prediction) in enumerate(zip(request.symbols, predictions)):
            results.append({
                "symbol": symbol,
                "prediction": float(prediction),
                "timestamp": datetime.now().isoformat()
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{symbol}")
async def get_predictions(symbol: str, limit: int = 100):
    """Get predictions for a symbol"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM predictions 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (symbol, limit))
                data = cur.fetchall()
        
        return {"symbol": symbol, "predictions": data, "count": len(data)}
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
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

@app.post("/models/deploy")
async def deploy_model(model_name: str, model_version: str):
    """Deploy a model version to production"""
    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        
        return {
            "message": f"Model {model_name} version {model_version} deployed to production",
            "model_name": model_name,
            "version": model_version,
            "stage": "Production"
        }
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
