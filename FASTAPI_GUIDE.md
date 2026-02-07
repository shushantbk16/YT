# FastAPI Complete Guide - AI/ML Edition

## What is FastAPI?

FastAPI is a modern Python web framework that makes it incredibly easy to build fast APIs (Application Programming Interfaces). It's perfect for serving machine learning models.

**Why FastAPI for ML/AI?**
- ⚡ **Super Fast** - One of the fastest Python frameworks
- 📚 **Auto Documentation** - Automatically generates interactive docs
- ✅ **Data Validation** - Built-in validation using Pydantic
- 🔄 **Async Support** - Handle thousands of concurrent requests
- 🚀 **Easy Deployment** - Works great with Docker & cloud services

---

## Installation

```bash
# Install FastAPI and Uvicorn (ASGI server)
pip install fastapi uvicorn

# For file upload support
pip install python-multipart

# For production
pip install gunicorn
```

---

## Core Concepts

### 1. Routes & HTTP Methods

```python
from fastapi import FastAPI

app = FastAPI()

# GET - Retrieve data
@app.get("/items/")
def read_items():
    return {"items": [...]}

# POST - Create/send data
@app.post("/items/")
def create_item(item: dict):
    return {"created": item}

# PUT - Update data
@app.put("/items/{item_id}")
def update_item(item_id: int, item: dict):
    return {"updated": item}

# DELETE - Remove data
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    return {"deleted": item_id}
```

### 2. Path Parameters vs Query Parameters

```python
# Path parameter - part of URL path
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}
# Usage: GET /items/5

# Query parameter - comes after ? in URL
@app.get("/search/")
def search(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
# Usage: GET /search/?skip=5&limit=20

# Both together
@app.get("/items/{item_id}")
def complex_endpoint(item_id: int, details: bool = False):
    return {"item_id": item_id, "details": details}
# Usage: GET /items/5?details=true
```

### 3. Request Body with Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    description: str = None

@app.post("/items/")
def create_item(item: Item):
    # FastAPI automatically validates the JSON
    # and converts it to Item object
    return {"received": item}

# Usage: POST /items/
# {
#   "name": "Laptop",
#   "price": 999.99,
#   "description": "High performance"
# }
```

---

## ML/AI Patterns

### Pattern 1: Simple Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict/")
def predict(input_data: PredictionInput):
    prediction = model.predict([input_data.features])
    return {
        "input": input_data.features,
        "prediction": float(prediction[0])
    }
```

### Pattern 2: Batch Predictions

```python
from typing import List

class BatchInput(BaseModel):
    samples: List[List[float]]

@app.post("/batch-predict/")
def batch_predict(batch: BatchInput):
    predictions = model.predict(batch.samples)
    return {
        "count": len(predictions),
        "predictions": predictions.tolist()
    }
```

### Pattern 3: Image Classification

```python
from fastapi import UploadFile, File
from PIL import Image
import io

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess and predict
    processed = preprocess(image)
    prediction = model.predict(processed)
    
    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": 0.95
    }
```

### Pattern 4: Text Analysis (NLP)

```python
class TextInput(BaseModel):
    text: str

@app.post("/sentiment/")
def analyze_sentiment(input_data: TextInput):
    # Use NLP model (BERT, etc.)
    sentiment = nlp_model(input_data.text)
    
    return {
        "text": input_data.text,
        "sentiment": sentiment["label"],
        "score": sentiment["score"]
    }
```

### Pattern 5: Authentication for APIs

```python
from fastapi import Header, HTTPException

VALID_API_KEYS = ["sk-1234567890"]

@app.post("/predict/")
async def predict(
    input_data: PredictionInput,
    api_key: str = Header(None)
):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Process prediction
    result = model.predict(input_data.features)
    return {"prediction": result}
```

---

## Running Your API

### Local Development

```bash
# Run with hot reload (changes auto-restart)
uvicorn main:app --reload

# Visit these URLs:
# - http://localhost:8000           → API root
# - http://localhost:8000/docs      → Interactive docs (Swagger UI)
# - http://localhost:8000/redoc     → Alternative docs
```

### Production Deployment

```bash
# Multiple workers for production
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

# Or with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Docker Deployment

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Testing Your API

### Using curl

```bash
# GET request
curl http://localhost:8000/items/5

# POST request with JSON
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.0, 2.0, 3.0]
  }'

# With authentication
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -H "api-key: sk-1234567890" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

### Using Python

```python
import requests

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict/",
    json={"features": [1.0, 2.0, 3.0]}
)

print(response.json())
```

### Using JavaScript

```javascript
// Fetch prediction
const response = await fetch('http://localhost:8000/predict/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    features: [1.0, 2.0, 3.0]
  })
});

const data = await response.json();
console.log(data);
```

---

## Real World Example: Iris Flower Classification

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = FastAPI(title="Iris Classifier")

# Load and train model
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/classify/")
def classify_iris(input_data: IrisInput):
    features = np.array([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        "flower_type": iris.target_names[prediction],
        "confidence": float(probability[prediction]),
        "probabilities": {
            iris.target_names[i]: float(prob)
            for i, prob in enumerate(probability)
        }
    }
```

**Test it:**
```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

---

## Common Mistakes to Avoid

❌ **Mistake 1:** Not validating input
```python
# Bad
@app.post("/predict/")
def predict(data):  # No type hints!
    return model.predict(data)
```

✅ **Good:**
```python
class Input(BaseModel):
    features: list[float]

@app.post("/predict/")
def predict(data: Input):
    return model.predict(data.features)
```

❌ **Mistake 2:** Blocking requests with long-running tasks
```python
# Bad - blocks other requests
@app.post("/train/")
def train_model(training_data: list):
    model.fit(training_data)  # Takes 1 hour!
    return {"status": "trained"}
```

✅ **Good:** Use async or background tasks
```python
from fastapi import BackgroundTasks

@app.post("/train/")
def train_model(
    training_data: list,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(model.fit, training_data)
    return {"status": "training..."}
```

---

## Next Steps

1. **Try the basic example** - Run `fastapi_learn.py`
2. **Try advanced features** - Run `advanced_fastapi_ml.py`
3. **Integrate your ML model** - Replace dummy models with real ones
4. **Add authentication** - Secure your API
5. **Deploy to cloud** - Use AWS, GCP, or Azure

---

## Resources

- [Official FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [Deploy to AWS](https://docs.aws.amazon.com/lambda/)
- [Deploy to Google Cloud](https://cloud.google.com/functions)
- [Docker Guide](https://docs.docker.com/)

---

## Quick Command Reference

```bash
# Install
pip install fastapi uvicorn

# Run (development)
uvicorn main:app --reload

# Run (production)
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/docs

# Install for file handling
pip install python-multipart

# View async/await patterns
# Search: "async def" in any FastAPI example
```

---

**You're ready to build AI/ML APIs with FastAPI!** 🚀
