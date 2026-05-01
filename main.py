from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# -----------------------------
# 1. Load Model
# -----------------------------
model = joblib.load("model/model.pkl")


# -----------------------------
# 4. Clean Text
# -----------------------------
def clean_text_series(text):
    return re.sub(r"http\S+", "", text)

# -----------------------------
# 2. Initialize App
# -----------------------------
app = FastAPI(title="Fake News Detection API")


# -----------------------------
# 3. Request Schema
# -----------------------------
class NewsRequest(BaseModel):
    text: str


# -----------------------------
# 4. Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}


@app.post("/predict")
def predict(data: NewsRequest):
    text = clean_text_series(data.text)

    prediction = model.predict([text])[0]

    result = "Fake" if prediction == 1 else "Real"

    return {"prediction": result}