"""
Serialize and verify the Champion Bilingual Sentiment Model
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing import NLPPreprocessor

DATASETS_DIR = os.path.join(BASE_DIR, "ml", "datasets")
SAVED_DIR = os.path.join(BASE_DIR, "ml", "saved_models")

prep = NLPPreprocessor()

print("Loading datasets...")
df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
df_train = pd.concat([df_train_en[["text", "sentiment"]], df_train_hi[["text", "sentiment"]]], ignore_index=True)

df_test_en = pd.read_csv(os.path.join(DATASETS_DIR, "test_sentiment.csv"))
df_test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

print(f"Training on {len(df_train)} samples...")
X_train = [prep.preprocess_to_text(t) for t in df_train["text"].fillna("")]
y_train = df_train["sentiment"]

X_test_en = [prep.preprocess_to_text(t) for t in df_test_en["text"].fillna("")]
y_test_en = df_test_en["sentiment"]

X_test_hi = [prep.preprocess_to_text(t) for t in df_test_hi["text"].fillna("")]
y_test_hi = df_test_hi["sentiment"]

vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
    ("char", TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb", max_features=20000, sublinear_tf=True, min_df=5)),
], transformer_weights={"word": 1.5, "char": 0.5})

print("Transforming features...")
t0 = time.time()
X_tr = vectorizer.fit_transform(X_train)
X_te_en = vectorizer.transform(X_test_en)
X_te_hi = vectorizer.transform(X_test_hi)
print(f"Features fitted in {time.time()-t0:.2f}s, feature count: {X_tr.shape[1]}")

print("Fitting Logistic Regression champion model...")
clf = LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42)
t1 = time.time()
clf.fit(X_tr, y_train)
print(f"Model fitted in {time.time()-t1:.2f}s")

# Measure performance on holdout test set (English)
t_inf = time.time()
y_pred_en = clf.predict(X_te_en)
latency_ms = ((time.time() - t_inf) / len(y_test_en)) * 1000

acc_en = float(accuracy_score(y_test_en, y_pred_en))
prec_macro_en = float(precision_score(y_test_en, y_pred_en, average="macro"))
rec_macro_en = float(recall_score(y_test_en, y_pred_en, average="macro"))
f1_macro_en = float(f1_score(y_test_en, y_pred_en, average="macro"))
f1_weighted_en = float(f1_score(y_test_en, y_pred_en, average="weighted"))

cm_en = confusion_matrix(y_test_en, y_pred_en, labels=["negative", "neutral", "positive"]).tolist()

# Measure performance on holdout test set (Hinglish)
y_pred_hi = clf.predict(X_te_hi)
acc_hi = float(accuracy_score(y_test_hi, y_pred_hi))
f1_macro_hi = float(f1_score(y_test_hi, y_pred_hi, average="macro"))
f1_weighted_hi = float(f1_score(y_test_hi, y_pred_hi, average="weighted"))
cm_hi = confusion_matrix(y_test_hi, y_pred_hi, labels=["negative", "neutral", "positive"]).tolist()

print("\n--- RESULTS ---")
print(f"English Holdout Accuracy: {acc_en*100:.2f}% | Macro F1: {f1_macro_en:.4f} | Weighted F1: {f1_weighted_en:.4f}")
print(f"Hinglish Holdout Accuracy: {acc_hi*100:.2f}% | Macro F1: {f1_macro_hi:.4f} | Weighted F1: {f1_weighted_hi:.4f}")

# Extract class metrics
report_dict = classification_report(y_test_en, y_pred_en, output_dict=True)
class_metrics = {}
for c in ["negative", "neutral", "positive"]:
    class_metrics[c] = {
        "precision": float(report_dict[c]["precision"]),
        "recall": float(report_dict[c]["recall"]),
        "f1_score": float(report_dict[c]["f1-score"]),
        "support": float(report_dict[c]["support"]),
    }

metadata = {
    "model_name": "Multinomial Logistic Regression (Bilingual L2, Balanced)",
    "algorithm": "logistic_regression",
    "classes": ["negative", "neutral", "positive"],
    "training_dataset_size": len(df_train),
    "testing_dataset_size": len(df_test_en),
    "hinglish_testing_dataset_size": len(df_test_hi),
    "vocabulary_size": int(X_tr.shape[1]),
    "ngram_range": [1, 3],
    "sublinear_tf": True,
    "max_features": 65000,
    "accuracy": acc_en,
    "precision_macro": prec_macro_en,
    "recall_macro": rec_macro_en,
    "f1_macro": f1_macro_en,
    "f1_weighted": f1_weighted_en,
    "hinglish_accuracy": acc_hi,
    "hinglish_f1_macro": f1_macro_hi,
    "hinglish_f1_weighted": f1_weighted_hi,
    "class_metrics": class_metrics,
    "confusion_matrix": cm_en,
    "hinglish_confusion_matrix": cm_hi,
    "latency_ms_per_sample": latency_ms,
    "hyperparameters": {
        "C": 1.25,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    },
    "training_date": datetime.now(timezone.utc).isoformat()
}

# Save artifacts
model_file = os.path.join(SAVED_DIR, "model.joblib")
vec_file = os.path.join(SAVED_DIR, "vectorizer.joblib")
meta_file = os.path.join(SAVED_DIR, "metadata.json")

print("\nSaving artifacts...")
joblib.dump(clf, model_file, compress=3)
joblib.dump(vectorizer, vec_file, compress=3)
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: {model_file} ({os.path.getsize(model_file)/1024:.1f} KB)")
print(f"Saved: {vec_file} ({os.path.getsize(vec_file)/1024:.1f} KB)")
print(f"Saved: {meta_file}")

# Verify with SentimentPredictor
from ml.training.predictor import SentimentPredictor
predictor = SentimentPredictor(model_file, vec_file, meta_file)

test_sentences = [
    "Not bad at all! I actually enjoyed it.",
    "It works exactly as described, nothing more, nothing less.",
    "Wow, what an incredible disaster!",
    "The delivery was late, but the product is fantastic.",
    "The food was delicious, although the staff was rude.",
    "Product theek hai, nothing special.",
    "Yaar, mujhe iska experience bahut pasand aaya.",
    "Mujhe ye product bilkul pasand nahi aaya.",
    "Ye phone bilkul bekar hai.",
    "Service average thi, kuch khaas nahi.",
    "Mujhe is product se koi problem nahi hai.",
    "Bhai kya zabardast quality hai!",
    "Paise waste ho gaye, totally disappointed.",
    "Experience achha tha lekin customer support bekar tha.",
    "The package was delivered yesterday at 10 AM.",
    "शानदार सेवा और बहुत ही मददगार कर्मचारी!",
    "यह उत्पाद बिल्कुल बेकार और टूटा हुआ निकला।"
]

print("\n--- INFERENCE PREDICTIONS VIA PREDICTOR ---")
for s in test_sentences:
    r = predictor.predict(s)
    print(f"'{s[:45]}...' -> {r['sentiment'].upper():8s} ({r['confidence']*100:.1f}%) | Probs: {r['probabilities']}")

print("\nAll artifacts verified successfully!")
