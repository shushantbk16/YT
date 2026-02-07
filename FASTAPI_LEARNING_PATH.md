# FastAPI Learning Path 🚀

## What You've Just Learned

You now have **4 complete FastAPI examples** to learn from:

### 1. **fastapi_learn.py** - Comprehensive Guide
- All basic concepts explained
- Simple examples with comments
- Perfect for understanding FastAPI fundamentals

### 2. **fastapi_quickstart.py** - Get Running Now! ⭐️
- **START HERE** - Simplest working example
- Iris flower classification
- Run: `uvicorn fastapi_quickstart:app --reload`
- Visit: `http://localhost:8000/docs`

### 3. **advanced_fastapi_ml.py** - Production Patterns
- Real-world ML patterns
- Authentication, caching, batch processing
- File uploads, async tasks
- Model versioning

### 4. **fastapi_production_structure.py** - How to Organize
- Professional project structure
- Configuration management
- Testing setup
- Docker deployment

---

## 🎯 Quick Start (5 Minutes)

```bash
# 1. Install
pip install fastapi uvicorn

# 2. Run the quickstart example
uvicorn fastapi_quickstart:app --reload

# 3. Open in browser
http://localhost:8000/docs

# 4. Try the endpoints!
```

**What you'll see:**
- Interactive API documentation (Swagger UI)
- All endpoints listed with examples
- "Try it out" button to test without coding

---

## 📚 Learning Progression

### Level 1: Basics (fastapi_quickstart.py)
Learn these concepts first:
- ✅ Routes (`@app.get()`, `@app.post()`)
- ✅ Request/Response models (Pydantic)
- ✅ Path parameters (`/predict/{id}`)
- ✅ Query parameters (`?skip=5&limit=10`)
- ✅ Auto documentation (`/docs`)

### Level 2: Intermediate (fastapi_learn.py)
Then learn:
- ✅ Error handling (`HTTPException`)
- ✅ Multiple endpoints
- ✅ Type hints and validation
- ✅ Response models
- ✅ API documentation

### Level 3: Advanced (advanced_fastapi_ml.py)
Finally learn:
- ✅ Authentication & API keys
- ✅ File uploads
- ✅ Caching results
- ✅ Batch processing
- ✅ Async operations
- ✅ Model versioning

### Level 4: Production (fastapi_production_structure.py)
Master:
- ✅ Project structure
- ✅ Configuration management
- ✅ Testing
- ✅ Docker deployment
- ✅ Logging

---

## 🤖 AI/ML Specific Examples

### Image Classification
```python
@app.post("/classify-image/")
def classify_image(image_url: str):
    result = model.predict(image_url)
    return {"prediction": result, "confidence": 0.95}
```

### Text Analysis (NLP)
```python
@app.post("/sentiment/")
def analyze_sentiment(text: str):
    sentiment = nlp_model(text)
    return {"text": text, "sentiment": sentiment}
```

### Model Serving
```python
@app.post("/predict/")
def predict(features: List[float]):
    result = model.predict([features])
    return {"prediction": result[0], "confidence": 0.92}
```

### Batch Predictions
```python
@app.post("/batch-predict/")
def batch_predict(samples: List[InputData]):
    predictions = [model.predict(s) for s in samples]
    return {"predictions": predictions}
```

---

## 🔧 Common Tasks

### Task 1: Serve a Trained Model

```python
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load your model
model = pickle.load(open("my_model.pkl", "rb"))

class Input(BaseModel):
    features: list[float]

@app.post("/predict/")
def predict(data: Input):
    result = model.predict([data.features])
    return {"prediction": float(result[0])}
```

### Task 2: Add Authentication

```python
from fastapi import Header, HTTPException

@app.post("/predict/")
def predict(data: Input, api_key: str = Header(...)):
    if api_key != "sk-1234567890":
        raise HTTPException(status_code=401, detail="Invalid key")
    
    result = model.predict([data.features])
    return {"prediction": result}
```

### Task 3: Handle File Uploads

```python
from fastapi import UploadFile, File
from PIL import Image
import io

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    result = model.predict(image)
    return {"prediction": result}
```

### Task 4: Process Multiple Items

```python
from typing import List

@app.post("/batch/")
def batch_process(items: List[Input]):
    results = []
    for item in items:
        pred = model.predict([item.features])
        results.append({"prediction": float(pred[0])})
    return {"results": results}
```

---

## 🚀 Deployment

### Local
```bash
uvicorn main:app --reload
```

### Production (4 workers)
```bash
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t my-api .
docker run -p 8000:8000 my-api
```

### Cloud Services
- **AWS Lambda**: Use API Gateway + Lambda
- **Google Cloud**: `gcloud run deploy`
- **Azure**: Azure App Service
- **Heroku**: `git push heroku main`

---

## 📊 API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Welcome/root |
| GET | `/docs` | Interactive docs |
| GET | `/redoc` | Alternative docs |
| GET | `/health` | Health check |
| POST | `/predict/` | Single prediction |
| POST | `/batch-predict/` | Multiple predictions |
| POST | `/upload/` | File upload |
| GET | `/models/` | List models |

---

## 💡 Tips & Best Practices

### ✅ DO:
- Use type hints for all parameters
- Create Pydantic models for requests/responses
- Handle errors with HTTPException
- Log important events
- Test your endpoints
- Use async for I/O operations

### ❌ DON'T:
- Don't skip input validation
- Don't block threads with long operations
- Don't expose sensitive info in docs
- Don't hardcode API keys
- Don't ignore error handling
- Don't leave debug=True in production

---

## 🧪 Testing Your API

### With curl
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

### With Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/",
    json={"features": [1.0, 2.0, 3.0]}
)
print(response.json())
```

### With JavaScript
```javascript
const response = await fetch('http://localhost:8000/predict/', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({features: [1.0, 2.0, 3.0]})
});
const data = await response.json();
```

---

## 📖 Resources

- [Official FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Server](https://www.uvicorn.org/)
- [Starlette Framework](https://www.starlette.io/)

---

## ❓ Common Questions

**Q: FastAPI vs Flask?**
A: FastAPI is faster, has automatic docs, and built-in validation. Flask is simpler for very basic APIs.

**Q: How to deploy to production?**
A: Docker + Docker Compose, Cloud Run, Lambda, or any Python hosting.

**Q: Can I use async/await?**
A: Yes! FastAPI supports async functions. Use `async def` instead of `def`.

**Q: How do I add authentication?**
A: Use Header parameters, OAuth2, or custom middleware.

**Q: Can I serve large models?**
A: Yes, but ensure your server has enough RAM. Use async for concurrent requests.

**Q: How do I handle file uploads?**
A: Use `UploadFile` parameter from `fastapi`.

---

## 🎓 Next Steps

1. ✅ Run `fastapi_quickstart.py`
2. ✅ Explore endpoints in `/docs`
3. ✅ Study `fastapi_learn.py` for concepts
4. ✅ Try `advanced_fastapi_ml.py` patterns
5. ✅ Build your own API with your ML model!

---

## 📝 Your First ML API Project

Here's a template to start:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="My ML API")

# Load your model
model = pickle.load(open("model.pkl", "rb"))

class Input(BaseModel):
    features: list[float]

class Output(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict/", response_model=Output)
def predict(data: Input):
    pred = model.predict([data.features])[0]
    confidence = model.predict_proba([data.features]).max()
    return Output(
        prediction=float(pred),
        confidence=float(confidence)
    )

@app.get("/health/")
def health():
    return {"status": "ok"}

# Run with: uvicorn main:app --reload
```

---

## 🎉 You're Ready!

You now understand FastAPI enough to:
- ✅ Build ML serving APIs
- ✅ Handle predictions
- ✅ Deploy to production
- ✅ Secure your endpoints
- ✅ Document automatically

**Start with `fastapi_quickstart.py` and run it now!**

Questions? Check the `/docs` endpoint or official FastAPI docs.

Happy building! 🚀
