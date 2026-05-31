import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

MODEL_PATH = "model.pkl"

def train_model():
    df = pd.read_csv("data/faults.csv")
    X = df["text"]
    y = df["severity"]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_model()

def predict_severity(text: str):
    model = load_model()
    return model.predict([text])[0]