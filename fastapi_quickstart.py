"""
FASTAPI QUICK START - Run this immediately!
=============================================

This is a simple, working FastAPI example you can test right now.

TO RUN:
    1. pip install fastapi uvicorn
    2. uvicorn fastapi_quickstart:app --reload
    3. Open http://localhost:8000/docs in browser
    4. Click "Try it out" to test endpoints
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import random

# Create the API
app = FastAPI(
    title="Simple ML API",
    description="Quick start FastAPI for ML",
    version="1.0"
)

# ============================================================
# DATA MODELS (How data looks when sent/received)
# ============================================================

class Iris(BaseModel):
    """Input data for iris prediction"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    """Response from prediction"""
    flower_type: str
    confidence: float
    input_data: Iris

# ============================================================
# FAKE MODEL TRAINING DATA
# ============================================================

# Simulated trained model
iris_models = {
    "setosa": {"ranges": [(4.3, 5.8), (2.3, 4.4), (1.0, 1.9), (0.1, 0.6)]},
    "versicolor": {"ranges": [(4.9, 7.0), (2.0, 3.4), (3.0, 5.1), (1.0, 1.8)]},
    "virginica": {"ranges": [(4.9, 7.9), (2.2, 3.8), (4.5, 6.9), (1.4, 2.5)]}
}

def predict_iris_flower(sepal_length, sepal_width, petal_length, petal_width):
    """
    Simple ML prediction function
    Determines iris flower type based on measurements
    """
    # Check which flower type best matches
    features = [sepal_length, sepal_width, petal_length, petal_width]
    best_match = None
    best_score = 0
    
    for flower_type, ranges_dict in iris_models.items():
        ranges = ranges_dict["ranges"]
        score = 0
        
        for i, (feature, (min_val, max_val)) in enumerate(zip(features, ranges)):
            if min_val <= feature <= max_val:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = flower_type
    
    confidence = best_score / 4.0  # 4 features total
    return best_match or "unknown", confidence

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Simple ML API",
        "endpoints": {
            "predict": "POST /predict/",
            "health": "GET /health/",
            "docs": "/docs"
        }
    }

@app.post("/predict/", response_model=PredictionResponse)
def predict(iris: Iris):
    """
    Predict iris flower type from measurements
    
    Example input:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    flower_type, confidence = predict_iris_flower(
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    )
    
    return PredictionResponse(
        flower_type=flower_type,
        confidence=confidence,
        input_data=iris
    )

@app.get("/health/")
def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_available": list(iris_models.keys())
    }

@app.get("/info/")
def get_info():
    """Get API information"""
    return {
        "api_name": "Simple ML API",
        "version": "1.0",
        "model": "Iris Flower Classifier",
        "model_accuracy": 0.95,
        "supported_flowers": list(iris_models.keys()),
        "features_required": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }

@app.post("/batch-predict/")
def batch_predict(flowers: List[Iris]):
    """
    Predict multiple flowers at once
    
    Example input:
    [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}
    ]
    """
    predictions = []
    
    for iris in flowers:
        flower_type, confidence = predict_iris_flower(
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width
        )
        predictions.append({
            "input": iris,
            "flower_type": flower_type,
            "confidence": confidence
        })
    
    return {
        "total_predictions": len(predictions),
        "predictions": predictions
    }

@app.get("/sample-data/")
def get_sample_data():
    """
    Get sample data for testing
    
    These are real iris measurements you can test with:
    - Setosa: small flowers with short petals
    - Versicolor: medium flowers
    - Virginica: large flowers with long petals
    """
    samples = {
        "setosa": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "versicolor": {
            "sepal_length": 5.9,
            "sepal_width": 3.0,
            "petal_length": 4.2,
            "petal_width": 1.5
        },
        "virginica": {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5
        }
    }
    return samples

# ============================================================
# ERROR HANDLING
# ============================================================

@app.get("/predict-random/")
def predict_random():
    """
    Generate random measurements and predict
    Useful for testing without typing data
    """
    random_iris = Iris(
        sepal_length=random.uniform(4, 8),
        sepal_width=random.uniform(2, 4.5),
        petal_length=random.uniform(1, 7),
        petal_width=random.uniform(0.1, 2.5)
    )
    
    flower_type, confidence = predict_iris_flower(
        random_iris.sepal_length,
        random_iris.sepal_width,
        random_iris.petal_length,
        random_iris.petal_width
    )
    
    return {
        "generated_measurement": random_iris,
        "prediction": flower_type,
        "confidence": confidence
    }

# ============================================================
# STATISTICS
# ============================================================

prediction_history = []

@app.post("/predict-and-log/")
def predict_and_log(iris: Iris):
    """Predict and save to history"""
    flower_type, confidence = predict_iris_flower(
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    )
    
    # Store in history
    prediction_history.append({
        "timestamp": datetime.now(),
        "input": iris.dict(),
        "prediction": flower_type,
        "confidence": confidence
    })
    
    return {
        "prediction": flower_type,
        "confidence": confidence,
        "total_predictions_made": len(prediction_history)
    }

@app.get("/history/")
def get_history():
    """Get all previous predictions"""
    return {
        "total": len(prediction_history),
        "history": prediction_history[-10:]  # Last 10 predictions
    }

# ============================================================
# RUN INSTRUCTIONS
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("FASTAPI QUICK START")
    print("=" * 60)
    print("\n1. Install: pip install fastapi uvicorn")
    print("2. Run: uvicorn fastapi_quickstart:app --reload")
    print("3. Visit: http://localhost:8000/docs")
    print("4. Try endpoints in the interactive interface")
    print("\n" + "=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
