"""
FASTAPI COMPLETE GUIDE - From Scratch to AI/ML Applications
============================================================

FastAPI is a modern, fast web framework for building APIs with Python.
It's perfect for serving ML models and building AI applications.

Key Features:
- Super fast (one of the fastest Python frameworks)
- Easy to learn and use
- Automatic interactive API documentation
- Built-in data validation
- Great for serving ML models
"""

# ============================================================
# 1. INSTALLATION & SETUP
# ============================================================
"""
Installation:
    pip install fastapi
    pip install uvicorn  # ASGI server to run FastAPI

Quick Start:
    uvicorn fastapi_learn:app --reload
    
Then visit: http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

# Create FastAPI application
app = FastAPI(title="ML & AI API", version="1.0.0")

# ============================================================
# 2. BASIC CONCEPTS
# ============================================================

# Define data models using Pydantic
class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

class Prediction(BaseModel):
    input_value: float
    predicted_value: float
    confidence: float

# ============================================================
# 3. SIMPLE ROUTES (HTTP METHODS)
# ============================================================

# GET - Retrieve data
@app.get("/")
def read_root():
    """Root endpoint - returns welcome message"""
    return {"message": "Welcome to FastAPI", "api_version": "1.0"}

@app.get("/hello/{name}")
def hello(name: str):
    """Path parameter - greet by name"""
    return {"message": f"Hello, {name}!"}

@app.get("/items/{item_id}")
def get_item(item_id: int, price: float = 10.0):
    """
    Path parameter: item_id (from URL path)
    Query parameter: price (from URL query string)
    
    Examples:
        /items/5?price=100
        /items/10?price=50.5
    """
    return {
        "item_id": item_id,
        "price": price,
        "total": item_id * price
    }

# POST - Create/send data
@app.post("/items/")
def create_item(item: Item):
    """
    Receives JSON data and auto-validates
    
    Example JSON body:
    {
        "name": "Laptop",
        "price": 999.99,
        "description": "High performance"
    }
    """
    return {
        "status": "created",
        "item": item,
        "item_total_price": item.price * 1.1  # 10% markup
    }

# PUT - Update data
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    """Update existing item"""
    return {
        "status": "updated",
        "item_id": item_id,
        "updated_item": item
    }

# DELETE - Remove data
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    """Delete item"""
    return {"status": "deleted", "item_id": item_id}

# ============================================================
# 4. QUERY PARAMETERS & VALIDATION
# ============================================================

@app.get("/search/")
def search_items(
    skip: int = 0,
    limit: int = 10,
    search_term: Optional[str] = None
):
    """
    Query parameters with defaults
    
    Example: /search/?skip=5&limit=20&search_term=laptop
    """
    return {
        "skip": skip,
        "limit": limit,
        "search_term": search_term,
        "message": f"Returning {limit} items starting from {skip}"
    }

# ============================================================
# 5. ERROR HANDLING
# ============================================================

@app.get("/user/{user_id}")
def get_user(user_id: int):
    """Demonstration of error handling"""
    if user_id < 1:
        raise HTTPException(
            status_code=400,
            detail="User ID must be greater than 0"
        )
    if user_id > 100:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    return {"user_id": user_id, "username": f"user_{user_id}"}

# ============================================================
# 6. AI/ML SPECIFIC EXAMPLES
# ============================================================

# Simulated ML Model
class SimpleLinearModel:
    """Mock ML Model - In reality this would be a trained model"""
    def __init__(self):
        self.slope = 2.5
        self.intercept = 10.0
    
    def predict(self, x):
        """y = mx + b"""
        return self.slope * x + self.intercept

# Initialize model
model = SimpleLinearModel()

@app.post("/predict/")
def predict(input_data: dict):
    """
    ML Prediction endpoint
    
    Example JSON:
    {
        "value": 5.0
    }
    """
    try:
        x = input_data.get("value")
        if x is None:
            raise ValueError("'value' field is required")
        
        prediction = model.predict(x)
        return {
            "input": x,
            "prediction": prediction,
            "model": "LinearModel v1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# 7. BATCH PREDICTIONS
# ============================================================

@app.post("/batch-predict/")
def batch_predict(values: List[float]):
    """
    Batch prediction - predict multiple values at once
    
    Example JSON:
    [1.0, 2.5, 5.0, 10.0]
    """
    predictions = [
        {"input": x, "prediction": model.predict(x)}
        for x in values
    ]
    return {
        "count": len(predictions),
        "predictions": predictions
    }

# ============================================================
# 8. SENTIMENT ANALYSIS SIMULATOR
# ============================================================

class TextInput(BaseModel):
    text: str

def analyze_sentiment(text: str):
    """Dummy sentiment analyzer"""
    positive_words = ["good", "great", "amazing", "perfect", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "worse"]
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return {"sentiment": "positive", "score": min(1.0, pos_count * 0.3)}
    elif neg_count > pos_count:
        return {"sentiment": "negative", "score": min(1.0, neg_count * 0.3)}
    else:
        return {"sentiment": "neutral", "score": 0.5}

@app.post("/sentiment/")
def sentiment_analysis(text_input: TextInput):
    """
    AI Sentiment Analysis
    
    Example JSON:
    {
        "text": "This product is amazing and wonderful!"
    }
    """
    result = analyze_sentiment(text_input.text)
    return {
        "text": text_input.text,
        "analysis": result
    }

# ============================================================
# 9. IMAGE CLASSIFICATION SIMULATOR
# ============================================================

class ImageData(BaseModel):
    image_url: str
    image_name: str

@app.post("/classify-image/")
def classify_image(image_data: ImageData):
    """
    Dummy image classification endpoint
    In real world, you'd use model like ResNet, YOLO, etc.
    
    Example JSON:
    {
        "image_url": "https://example.com/dog.jpg",
        "image_name": "my_dog"
    }
    """
    # Simulated classification
    classes = ["dog", "cat", "bird"]
    confidences = [0.85, 0.10, 0.05]
    
    return {
        "image": image_data.image_name,
        "classifications": [
            {"class": cls, "confidence": conf}
            for cls, conf in zip(classes, confidences)
        ],
        "top_prediction": classes[0]
    }

# ============================================================
# 10. MODEL INFO & HEALTH CHECK
# ============================================================

@app.get("/model-info/")
def get_model_info():
    """Get information about current model"""
    return {
        "model_name": "LinearModel",
        "version": "1.0",
        "input_features": ["numerical_value"],
        "output": "prediction",
        "accuracy": 0.87,
        "model_type": "regression"
    }

@app.get("/health/")
def health_check():
    """Health check endpoint - used by load balancers"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "api_version": "1.0"
    }

# ============================================================
# 11. DOCUMENTATION ENDPOINTS
# ============================================================

@app.get("/api-docs/")
def api_documentation():
    """Return API documentation"""
    return {
        "endpoints": {
            "GET /": "Welcome message",
            "GET /hello/{name}": "Greet by name",
            "POST /predict/": "Get prediction from ML model",
            "POST /batch-predict/": "Predict multiple values",
            "POST /sentiment/": "Analyze sentiment",
            "POST /classify-image/": "Classify image",
            "GET /model-info/": "Get model information",
            "GET /health/": "Health check"
        },
        "docs_auto": "Visit /docs for interactive documentation"
    }

# ============================================================
# SECTION: RUNNING THE APP
# ============================================================

"""
HOW TO RUN:

1. Install dependencies:
    pip install fastapi uvicorn

2. Run the API:
    uvicorn fastapi_learn:app --reload

3. Access the API:
    - Browser: http://localhost:8000
    - Interactive Docs: http://localhost:8000/docs (Swagger UI)
    - Alternative Docs: http://localhost:8000/redoc

4. Test endpoints:
    curl http://localhost:8000/hello/Alice
    curl -X POST http://localhost:8000/predict/ -H "Content-Type: application/json" -d '{"value": 5.0}'
    curl -X POST http://localhost:8000/sentiment/ -H "Content-Type: application/json" -d '{"text": "This is amazing"}'
"""

# ============================================================
# KEY ADVANTAGES FOR AI/ML
# ============================================================

"""
Why FastAPI is Perfect for ML/AI:

1. FAST PERFORMANCE
   - Serves predictions with minimal latency
   - Handles concurrent requests efficiently

2. AUTOMATIC VALIDATION
   - Pydantic validates input data
   - Catches errors before reaching model

3. DOCUMENTATION
   - Auto-generates interactive docs
   - /docs shows all endpoints with examples

4. TYPE HINTS
   - Python type hints make code clear
   - IDE autocomplete works perfectly

5. EASY DEPLOYMENT
   - Works with Docker
   - Perfect for cloud services (AWS, GCP, Azure)

6. PRODUCTION-READY
   - Error handling built-in
   - Request logging available
   - Can run with multiple workers

7. MIDDLEWARE SUPPORT
   - Add authentication, CORS, logging
   - Pre/post-process requests
"""

# ============================================================
# COMPARISON WITH OTHER FRAMEWORKS
# ============================================================

"""
FastAPI vs Flask vs Django:

Feature          | FastAPI | Flask | Django
-----------------+---------+-------+--------
Speed            | Very Fast| Medium| Slow
Learning Curve   | Easy    | Easy  | Hard
Documentation    | Auto    | Manual| Auto
Validation       | Built-in| None  | Built-in
Type Hints       | Yes     | No    | Limited
Async Support    | Yes     | Limited| Yes
Best For         | APIs    | APIs  | Full Apps
ML Serving       | ★★★★★  | ★★★  | ★★
"""

# ============================================================
# COMMON ML/AI PATTERNS
# ============================================================

"""
1. SERVING PRE-TRAINED MODELS
   @app.post("/predict/")
   def predict(data: InputData):
       result = model.predict(data.features)
       return {"prediction": result}

2. BATCH PROCESSING
   @app.post("/batch-predict/")
   def batch_predict(items: List[InputData]):
       results = [model.predict(item) for item in items]
       return {"predictions": results}

3. FEATURE EXTRACTION
   @app.post("/extract-features/")
   def extract_features(image_data: bytes):
       features = feature_extractor(image_data)
       return {"features": features}

4. MODEL RETRAINING
   @app.post("/retrain/")
   def retrain_model(training_data: List[dict]):
       model.train(training_data)
       return {"status": "retrained"}

5. A/B TESTING
   @app.post("/predict-ab/")
   def predict_ab_test(data: InputData, model_version: str = "v1"):
       model = get_model(model_version)
       return model.predict(data)
"""

print("=" * 60)
print("FastAPI Learning Guide Created!")
print("=" * 60)
print("\nTo run this API:")
print("1. pip install fastapi uvicorn")
print("2. uvicorn fastapi_learn:app --reload")
print("3. Visit http://localhost:8000/docs")
print("=" * 60)
