# -----------------------------
# 1. Imports
# -----------------------------
import re
import joblib
import pandas as pd

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# 2. Load Dataset
# -----------------------------
def load_data(fake_path, true_path):
    print("2) Data loaded...")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# -----------------------------
# 3. Preprocess (cleaning here)
# -----------------------------
def preprocess(df):
    print("3) Preprocess done...")

    df["content"] = df["title"] + " " + df["text"]

    df["content"] = df["content"].apply(clean_text_series)

    return df["content"], df["label"]


# -----------------------------
# 4. Clean Text (for Pipeline)
# -----------------------------
def clean_text_series(text):
    return re.sub(r"http\S+", "", text)


# -----------------------------
# 5. Train Model
# -----------------------------
def train_model(X_train, y_train):
    print("4) Model training started...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("svm", LinearSVC(C=4))
    ])

    model.fit(X_train, y_train)
    return model


# -----------------------------
# 6. Evaluate Model
# -----------------------------
def evaluate(model, X_train, X_test, y_train, y_test):
    print("5) Evaluation...\n")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy train vs test (check overfitting or underfitting)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy : {test_acc}")

    # Confusion Matrix
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))


# -----------------------------
# 7. Save Model
# -----------------------------
def save_model(model, path="model/model.pkl"):
    joblib.dump(model, path)
    print(f"\n✅ Model saved at: {path}")


# -----------------------------
# 8. Main Function
# -----------------------------
def main():
    print("1) Started...")

    df = load_data("data/Fake.csv", "data/True.csv")

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate(model, X_train, X_test, y_train, y_test)

    save_model(model)

    print("\nCompleted all steps...\n")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()