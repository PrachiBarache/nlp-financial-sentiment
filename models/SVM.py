import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.preprocessing import clean_text

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class SVMModel:
    def __init__(self, model_path="models/svm.pkl", vectorizer_path="models/tfidf.pkl"):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str):
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(X)[0]
        label_idx = np.argmax(probs)
        return LABEL_MAP[label_idx], float(probs[label_idx])
