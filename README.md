# 📰 Fake News Detection API

A Machine Learning-based **Fake News Detection system** built using **TF-IDF + Linear SVM**, exposed via a **FastAPI backend** and deployed on **Render**.

---

## 🚀 Live API

🔗 **Base URL:**
`https://your-app-name.onrender.com`

📌 **Prediction Endpoint:**
`POST /predict`

---

## 📌 Features

* 🔍 Detects whether a news article is **Fake** or **Real**
* ⚡ Fast inference using **LinearSVC**
* 🧠 Text vectorization with **TF-IDF (uni + bi-grams)**
* 🧩 End-to-end pipeline (preprocessing + model)
* 🌐 REST API using **FastAPI**
* ☁️ Deployed on **Render**

---

## 🏗️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* Uvicorn
* Pandas
* Joblib

---

## 📂 Project Structure

```
fake-news-detection/
│
├── data/                # Dataset folder (not included)
├── model/               # Saved model (model.pkl)
│
├── notebooks/           # EDA & evaluation notebooks
│   ├── eda.ipynb
│   └── evaluation.ipynb
│
├── main.py              # FastAPI application
├── train.py             # Model training script
├── test_api.py          # API testing script
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone repository

```
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

### 2. Create virtual environment (optional but recommended)

```
python -m venv myvenv
source myvenv/bin/activate   # Linux/Mac
myvenv\Scripts\activate      # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start FastAPI server:

```
uvicorn main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Usage

### Endpoint:

```
POST /predict
```

### Request Body:

```json
{
  "text": "Breaking news: government releases new policy"
}
```

### Response:

```json
{
  "prediction": "Real"
}
```

---

## 🧪 Testing API

You can test using:

### Python script:

```
python test_api.py
```

### Or using curl:

```
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Shocking truth they don\'t want you to know!!!"}'
```

---

## 🧠 Model Details

* Algorithm: **Linear Support Vector Machine (LinearSVC)**
* Vectorizer: **TF-IDF**

  * max_features = 5000
  * ngram_range = (1, 2)
* Final Hyperparameter:

  * **C = 4 (optimized using cross-validation & F1-score)**

---

## 📊 Performance

* Accuracy: **~99.5%**
* Balanced precision and recall
* Minimal overfitting (validated using train vs test comparison)

---

## 📁 Dataset

Place dataset files inside `data/` folder:

* `Fake.csv`
* `True.csv`

> Dataset not included due to size. You can use publicly available fake news datasets.

---

## ☁️ Deployment

This project is deployed using **Render**.

### Start Command:

```
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

## 🤝 Contributing

Feel free to fork the repo and improve it!

---

## 👨‍💻 Author

**Anjan Pal**

- 🔗 LinkedIn: https://www.linkedin.com/in/anjan-pal-ab5a5a247
- 💼 Open to opportunities in Machine Learning / Python Development

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
