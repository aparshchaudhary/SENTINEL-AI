"""
SENTINEL AI – Reusable Sentiment Inference Engine
Academic Project: CSE276 – Artificial Intelligence Foundations

Loads serialized TF-IDF vectorizer and trained classifier artifacts,
executes identical preprocessing, and produces real-time class predictions
and probability distributions.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing import NLPPreprocessor


class SentimentPredictor:
    """
    Production-grade inference engine for SENTINEL AI.
    Integrates NLP preprocessing, TF-IDF vector transformation,
    and supervised classification with probability estimation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        saved_dir = os.path.join(BASE_DIR, "ml", "saved_models")
        self.model_path = model_path or os.path.join(saved_dir, "model.joblib")
        self.vectorizer_path = vectorizer_path or os.path.join(saved_dir, "vectorizer.joblib")
        self.metadata_path = metadata_path or os.path.join(saved_dir, "metadata.json")

        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.classes: List[str] = ["negative", "neutral", "positive"]

        # Deterministic preprocessing identical to training pipeline
        self.preprocessor = NLPPreprocessor()
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, vectorizer, and metadata from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer artifact not found at: {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                if "classes" in self.metadata:
                    self.classes = self.metadata["classes"]
        else:
            self.classes = list(getattr(self.model, "classes_", ["negative", "neutral", "positive"]))

    def predict(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Predicts sentiment polarity and estimated class probabilities for a single text input.

        Args:
            text: Raw input string.

        Returns:
            Dictionary matching the API/service contract:
            {
                "text": str,
                "cleaned_text": str,
                "sentiment": "positive" | "negative" | "neutral",
                "confidence": float (maximum estimated class probability),
                "probabilities": { "negative": float, "neutral": float, "positive": float },
                "model_name": str,
                "is_empty": bool
            }
        """
        # Safe input sanitization
        if text is None or not isinstance(text, str) or text.strip() == "":
            return {
                "text": text if text is not None else "",
                "cleaned_text": "",
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {cls: 0.0 for cls in self.classes},
                "model_name": self.metadata.get("model_name", "Trained Classifier") if self.metadata else "Model",
                "is_empty": True,
            }

        # 1. Apply identical preprocessing
        cleaned_text = self.preprocessor.preprocess_to_text(text)

        # Handle degenerate cases where cleaning strips all characters (e.g. "?!@#$%" or standalone URLs)
        if not cleaned_text.strip():
            return {
                "text": text,
                "cleaned_text": "",
                "sentiment": "neutral",
                "confidence": 0.3333,
                "probabilities": {cls: round(1.0 / len(self.classes), 4) for cls in self.classes},
                "model_name": self.metadata.get("model_name", "Trained Classifier") if self.metadata else "Model",
                "is_empty": False,
            }

        # 2. Vectorize with fitted TF-IDF vocabulary
        features = self.vectorizer.transform([cleaned_text])

        # 3. Model inference: prediction and probability estimation
        predicted_sentiment = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]

        # Model classes mapping
        model_classes = list(self.model.classes_)
        probabilities = {
            cls: round(float(probs[model_classes.index(cls)]), 4) for cls in self.classes
        }

        # Maximum posterior probability as estimated prediction confidence
        confidence = round(float(np.max(probs)), 4)

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": str(predicted_sentiment),
            "confidence": confidence,
            "probabilities": probabilities,
            "model_name": self.metadata.get("model_name", "Trained Classifier") if self.metadata else "Model",
            "is_empty": False,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Efficient vectorized batch inference for list of input texts.
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
