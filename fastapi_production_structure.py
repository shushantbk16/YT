"""
FASTAPI PRODUCTION STRUCTURE - Best Practices
==============================================

This shows how to organize a real ML/AI project with FastAPI

Project Structure:
    ml_api/
    ├── main.py                  # Main application
    ├── config.py               # Configuration
    ├── models/
    │   ├── __init__.py
    │   └── iris_model.pkl      # Trained model
    ├── schemas/
    │   └── prediction.py       # Request/Response models
    ├── api/
    │   ├── __init__.py
    │   ├── routes.py          # API endpoints
    │   └── dependencies.py    # Shared dependencies
    ├── ml/
    │   ├── __init__.py
    │   └── predictor.py       # ML logic
    └── requirements.txt        # Dependencies
"""

# ============================================================
# 1. CONFIG.py - Configuration Settings
# ============================================================

"""
config.py content:

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "ML Prediction API"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", False)
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/iris_model.pkl")
    MODEL_VERSION = "1.0"
    
    # API settings
    API_WORKERS = int(os.getenv("WORKERS", 4))
    
    # Database (if needed)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

settings = Settings()
"""

# ============================================================
# 2. SCHEMAS.py - Data Models
# ============================================================

"""
schemas/prediction.py content:

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=8)
    sepal_width: float = Field(..., gt=0, le=5)
    petal_length: float = Field(..., gt=0, le=7)
    petal_width: float = Field(..., gt=0, le=3)

class PredictionOutput(BaseModel):
    flower_type: str
    confidence: float
    input_data: IrisInput
    model_version: str
    timestamp: datetime

class BatchPredictionInput(BaseModel):
    samples: List[IrisInput] = Field(..., min_items=1, max_items=1000)

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total: int
    successful: int
    failed: int
"""

# ============================================================
# 3. ML/PREDICTOR.py - ML Logic
# ============================================================

"""
ml/predictor.py content:

import pickle
from typing import Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class IrisPredictor:
    def __init__(self, model_path: str):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess(self, features: list) -> list:
        # Normalize features if needed
        return features
    
    def predict(self, features: list) -> Tuple[str, float]:
        try:
            processed = self.preprocess(features)
            prediction = self.model.predict([processed])[0]
            confidence = self.model.predict_proba([processed]).max()
            return prediction, confidence
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

# Global predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        from config import settings
        predictor = IrisPredictor(settings.MODEL_PATH)
    return predictor
"""

# ============================================================
# 4. API/DEPENDENCIES.py - Shared Dependencies
# ============================================================

"""
api/dependencies.py content:

from fastapi import Depends, HTTPException
from ml.predictor import get_predictor

async def verify_api_key(api_key: str = Header(None)) -> str:
    # Check API key
    valid_keys = ["sk-key-123", "sk-key-456"]
    if api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

async def get_predictor_service():
    return get_predictor()
"""

# ============================================================
# 5. API/ROUTES.py - Endpoints
# ============================================================

"""
api/routes.py content:

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime
from schemas.prediction import (
    IrisInput, PredictionOutput, 
    BatchPredictionInput, BatchPredictionOutput
)
from ml.predictor import IrisPredictor, get_predictor

router = APIRouter(prefix="/api/v1", tags=["predictions"])

@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: IrisInput,
    predictor: IrisPredictor = Depends(get_predictor)
):
    try:
        features = [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]
        
        prediction, confidence = predictor.predict(features)
        
        return PredictionOutput(
            flower_type=prediction,
            confidence=float(confidence),
            input_data=input_data,
            model_version="1.0",
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict", response_model=BatchPredictionOutput)
async def batch_predict(
    batch: BatchPredictionInput,
    predictor: IrisPredictor = Depends(get_predictor)
):
    predictions = []
    successful = 0
    failed = 0
    
    for input_data in batch.samples:
        try:
            features = [
                input_data.sepal_length,
                input_data.sepal_width,
                input_data.petal_length,
                input_data.petal_width
            ]
            
            prediction, confidence = predictor.predict(features)
            
            predictions.append(PredictionOutput(
                flower_type=prediction,
                confidence=float(confidence),
                input_data=input_data,
                model_version="1.0",
                timestamp=datetime.now()
            ))
            successful += 1
        except Exception as e:
            failed += 1
    
    return BatchPredictionOutput(
        predictions=predictions,
        total=len(batch.samples),
        successful=successful,
        failed=failed
    )
"""

# ============================================================
# 6. MAIN.py - Application Entry Point
# ============================================================

"""
main.py content:

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import settings
from api.routes import router

# Create app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=settings.API_WORKERS
    )
"""

# ============================================================
# 7. REQUIREMENTS.txt
# ============================================================

"""
requirements.txt content:

fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3
python-multipart==0.0.6
python-dotenv==1.0.0
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.3
"""

# ============================================================
# 8. .ENV - Environment Variables
# ============================================================

"""
.env content:

DEBUG=True
MODEL_PATH=./models/iris_model.pkl
WORKERS=4
DATABASE_URL=sqlite:///./test.db
"""

# ============================================================
# 9. DOCKER - Containerization
# ============================================================

"""
Dockerfile content:

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

"""
docker-compose.yml content:

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - WORKERS=4
    volumes:
      - ./models:/app/models
"""

# ============================================================
# 10. TESTING
# ============================================================

"""
test_main.py content:

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/api/v1/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "flower_type" in response.json()

def test_batch_predict():
    response = client.post("/api/v1/batch-predict", json={
        "samples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}
        ]
    })
    assert response.status_code == 200
    assert response.json()["total"] == 2

# Run tests: pytest test_main.py
"""

# ============================================================
# DEPLOYMENT COMMANDS
# ============================================================

"""
DEPLOYMENT STEPS:

1. Local Development:
   uvicorn main:app --reload

2. Production with Gunicorn:
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

3. Docker Build:
   docker build -t ml-api .

4. Docker Run:
   docker run -p 8000:8000 ml-api

5. Docker Compose:
   docker-compose up

6. AWS Lambda:
   - Use AWS Lambda with API Gateway
   - Configure CloudWatch for monitoring

7. Google Cloud Run:
   gcloud run deploy ml-api --source .

8. Heroku:
   heroku create my-ml-api
   git push heroku main
"""

# ============================================================
# MONITORING & LOGGING
# ============================================================

"""
logging_config.py content:

import logging
import logging.handlers

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        'app.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
"""

print("=" * 60)
print("Production FastAPI Structure Guide")
print("=" * 60)
print("\nKey Principles:")
print("✓ Models are separate from routes")
print("✓ Configuration is centralized")
print("✓ Schemas define all data formats")
print("✓ Dependencies are managed properly")
print("✓ Error handling is comprehensive")
print("✓ Logging is configured")
print("✓ Docker-ready for deployment")
print("=" * 60)
