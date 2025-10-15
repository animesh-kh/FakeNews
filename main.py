from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI(title="Fake News Detector API")

# Load model & vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Request body schema
class NewsItem(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the Fake News Detection API 🧠"}

@app.post("/predict")
def predict(news: NewsItem):
    text = news.text
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    probs = model.predict_proba(transformed)
    label = "FAKE" if prediction == 0 else "REAL"

    return {"prediction": label,"confidence": max(probs[0])}
