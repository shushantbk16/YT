"""
ADVANCED FastAPI FOR AI/ML - Real-World Examples
================================================

This file shows production-ready FastAPI patterns for ML applications
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
import time

app = FastAPI(
    title="Advanced ML API",
    description="Production-ready ML serving with FastAPI",
    version="2.0.0"
)

# ============================================================
# 1. ADVANCED REQUEST MODELS WITH VALIDATION
# ============================================================

class ImagePredictionRequest(BaseModel):
    """Request model with validation"""
    image_path: str = Field(..., min_length=1, description="Path to image")
    model_version: str = Field(default="v1.0", description="Model version to use")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    
    @validator('confidence_threshold')
    def validate_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v

class ModelMetrics(BaseModel):
    """Response model with metrics"""
    prediction: str
    confidence: float
    processing_time_ms: float
    model_version: str
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    images: List[str] = Field(..., min_items=1, max_items=100)
    model_version: str = "v1.0"

# ============================================================
# 2. SIMULATED ML MODELS
# ============================================================

class ImageClassifier:
    """Simulated image classifier"""
    def __init__(self):
        self.classes = ["cat", "dog", "bird", "fish"]
        self.version = "1.0"
        self.load_time = 0.1
    
    def predict(self, image_path: str, threshold: float = 0.5):
        """Predict image class"""
        # Simulate model processing
        time.sleep(0.05)
        
        # Simulated predictions
        predictions = {
            "cat": 0.85,
            "dog": 0.10,
            "bird": 0.04,
            "fish": 0.01
        }
        
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]
        
        if confidence < threshold:
            return None, confidence
        
        return top_class, confidence

classifier = ImageClassifier()

# ============================================================
# 3. RESPONSE MODELS
# ============================================================

@app.post("/predict", response_model=ModelMetrics)
async def predict_image(request: ImagePredictionRequest):
    """
    Predict image class with full metrics
    
    Request body:
    {
        "image_path": "/images/cat.jpg",
        "model_version": "v1.0",
        "confidence_threshold": 0.5
    }
    """
    start_time = time.time()
    
    try:
        # Validate image path
        if not request.image_path.endswith(('.jpg', '.png', '.jpeg')):
            raise HTTPException(
                status_code=400,
                detail="Only .jpg, .png, .jpeg files are supported"
            )
        
        # Get prediction
        prediction, confidence = classifier.predict(
            request.image_path,
            request.confidence_threshold
        )
        
        if prediction is None:
            raise HTTPException(
                status_code=400,
                detail=f"Confidence {confidence:.2f} below threshold {request.confidence_threshold}"
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ModelMetrics(
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time,
            model_version=request.model_version,
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 4. BATCH PROCESSING WITH STREAMING
# ============================================================

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction with progress"""
    results = []
    
    for idx, image_path in enumerate(request.images):
        try:
            prediction, confidence = classifier.predict(image_path)
            results.append({
                "image": image_path,
                "prediction": prediction,
                "confidence": float(confidence),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "image": image_path,
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "total": len(request.images),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
        "processing_timestamp": datetime.now()
    }

# ============================================================
# 5. AUTHENTICATION & HEADERS
# ============================================================

@app.post("/predict-authenticated")
async def predict_with_auth(
    request: ImagePredictionRequest,
    api_key: Optional[str] = Header(None)
):
    """Prediction endpoint with API key authentication"""
    
    # Simple API key validation
    VALID_KEYS = ["sk-1234567890", "sk-0987654321"]
    
    if not api_key or api_key not in VALID_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    
    # Process request
    prediction, confidence = classifier.predict(request.image_path)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "authenticated": True
    }

# ============================================================
# 6. CACHING RESULTS
# ============================================================

# Simple in-memory cache for predictions
prediction_cache: Dict[str, dict] = {}

@app.post("/predict-cached")
async def predict_cached(request: ImagePredictionRequest):
    """Prediction with caching"""
    
    cache_key = f"{request.image_path}_{request.model_version}"
    
    # Check cache
    if cache_key in prediction_cache:
        return {
            "prediction": prediction_cache[cache_key],
            "from_cache": True
        }
    
    # New prediction
    start_time = time.time()
    prediction, confidence = classifier.predict(request.image_path)
    processing_time = time.time() - start_time
    
    result = {
        "prediction": prediction,
        "confidence": float(confidence),
        "processing_time": processing_time,
        "from_cache": False
    }
    
    # Store in cache
    prediction_cache[cache_key] = result
    
    return result

# ============================================================
# 7. MODEL MANAGEMENT
# ============================================================

class ModelInfo(BaseModel):
    name: str
    version: str
    accuracy: float
    latency_ms: float
    classes: List[str]

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    return [
        ModelInfo(
            name="ImageClassifier",
            version="1.0",
            accuracy=0.94,
            latency_ms=50,
            classes=classifier.classes
        ),
        ModelInfo(
            name="ImageClassifier",
            version="2.0",
            accuracy=0.96,
            latency_ms=60,
            classes=classifier.classes
        )
    ]

@app.post("/load-model/{model_version}")
async def load_model(model_version: str):
    """Load a specific model version"""
    valid_versions = ["1.0", "2.0"]
    
    if model_version not in valid_versions:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {model_version} not found"
        )
    
    return {
        "status": "loaded",
        "model": "ImageClassifier",
        "version": model_version,
        "timestamp": datetime.now()
    }

# ============================================================
# 8. FILE UPLOAD & PROCESSING
# ============================================================

@app.post("/upload-and-predict")
async def upload_and_predict(file: UploadFile = File(...)):
    """Upload image and get prediction"""
    
    # Validate file type
    valid_types = ["image/jpeg", "image/png"]
    
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )
    
    # In production, save file and process
    # For now, just return mock prediction
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "prediction": "cat",
        "confidence": 0.89,
        "message": "File received and processed"
    }

# ============================================================
# 9. ASYNC ENDPOINTS FOR LONG-RUNNING TASKS
# ============================================================

# Simulated task queue
task_results: Dict[str, dict] = {}

@app.post("/async-predict/{task_id}")
async def async_predict(task_id: str, request: ImagePredictionRequest):
    """Start async prediction task"""
    
    # In production, use Celery, RQ, or similar
    prediction, confidence = classifier.predict(request.image_path)
    
    task_results[task_id] = {
        "task_id": task_id,
        "status": "completed",
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.now()
    }
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "message": "Prediction task submitted"
    }

@app.get("/async-predict/{task_id}")
async def get_async_result(task_id: str):
    """Get result of async task"""
    
    if task_id not in task_results:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return task_results[task_id]

# ============================================================
# 10. HEALTH & PERFORMANCE MONITORING
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": 1,
        "api_version": "2.0"
    }

@app.get("/stats")
async def get_stats():
    """API statistics"""
    return {
        "total_predictions": len(prediction_cache),
        "cached_results": len(prediction_cache),
        "available_models": 2,
        "uptime_hours": 24
    }

# ============================================================
# DEPLOYMENT INSTRUCTIONS
# ============================================================

"""
RUNNING THIS API:

1. Install dependencies:
   pip install fastapi uvicorn python-multipart

2. Run with hot reload:
   uvicorn advanced_fastapi_ml:app --reload

3. Run in production (multiple workers):
   uvicorn advanced_fastapi_ml:app --workers 4 --host 0.0.0.0 --port 8000

4. Access:
   - API: http://localhost:8000/predict
   - Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - OpenAPI JSON: http://localhost:8000/openapi.json

TESTING WITH CURL:

# Simple prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/images/cat.jpg",
    "model_version": "v1.0",
    "confidence_threshold": 0.5
  }'

# Authentication
curl -X POST "http://localhost:8000/predict-authenticated" \
  -H "Content-Type: application/json" \
  -H "api-key: sk-1234567890" \
  -d '{
    "image_path": "/images/dog.jpg",
    "model_version": "v1.0"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["/images/cat.jpg", "/images/dog.jpg"],
    "model_version": "v1.0"
  }'

# List models
curl http://localhost:8000/models

# Health check
curl http://localhost:8000/health
"""

print("=" * 60)
print("Advanced FastAPI ML/AI Examples Ready!")
print("=" * 60)
