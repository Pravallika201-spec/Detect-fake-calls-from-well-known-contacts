"""NLP behavior and intent analysis models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.config.settings import MODEL_DIR


@dataclass
class IntentScore:
    text_window: str
    scam_probability: float
    triggers: List[str]


RISK_KEYWORDS = [
    "otp", "wire transfer", "gift card", "urgent", "arrest", "fine", "bank", "password",
]


class BehaviorClassifier:
    def __init__(self):
        self.vectorizer_path = MODEL_DIR / "intent_vectorizer.joblib"
        self.classifier_path = MODEL_DIR / "intent_classifier.joblib"
        if self.vectorizer_path.exists() and self.classifier_path.exists():
            self.vectorizer: TfidfVectorizer = joblib.load(self.vectorizer_path)
            self.model: LogisticRegression = joblib.load(self.classifier_path)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            self.model = LogisticRegression()

    def fit(self, texts: List[str], labels: List[int]) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.model, self.classifier_path)

    def predict(self, transcript: List[str]) -> List[IntentScore]:
        if not transcript:
            return []
        joined = [" ".join(transcript)]
        X = self.vectorizer.transform(joined)
        prob = float(self.model.predict_proba(X)[0][1]) if hasattr(self.model, "predict_proba") else 0.0
        triggered = [kw for kw in RISK_KEYWORDS if kw in joined[0].lower()]
        return [IntentScore(text_window=joined[0], scam_probability=prob, triggers=triggered)]
