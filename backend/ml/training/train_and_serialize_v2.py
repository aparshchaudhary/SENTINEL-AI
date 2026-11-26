"""
SENTINEL AI: Train and Serialize Champion Model v2 (Ingested Curated Dataset)
- Backs up previous champion v1
- Ingests non-overlapping curated bilingual dataset
- Evaluates on English and Hinglish holdout sets
- Serializes model, vectorizer, and metadata
"""

import os
import sys
import shutil
import json
import time
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing import NLPPreprocessor

DATASETS_DIR = os.path.join(BASE_DIR, "ml", "datasets")
SAVED_DIR = os.path.join(BASE_DIR, "ml", "saved_models")

# Step 1: Backup current champion v1
v1_dir = os.path.join(SAVED_DIR, "champion_v1")
os.makedirs(v1_dir, exist_ok=True)
for fname in ["model.joblib", "vectorizer.joblib", "metadata.json"]:
    src = os.path.join(SAVED_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(v1_dir, fname))
print(f"Backed up champion v1 to {v1_dir}")

# Step 2: Load datasets
df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
df_test_en = pd.read_csv(os.path.join(DATASETS_DIR, "test_sentiment.csv"))
df_test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

inc_df = pd.read_csv(os.path.join(DATASETS_DIR, "new_sentiment_incoming.tsv"), sep="\t")
inc_df["sentiment"] = inc_df["sentiment"].str.strip().str.lower()
inc_df["language"] = inc_df["language"].replace({"En": "English"}).fillna("English")

# Filter for curated rows
inc_curated = inc_df[inc_df["id"].str.slice(4).astype(int) <= 270].copy()

# Leakage check against both test sets
def norm(t): return re.sub(r"\s+", " ", str(t).strip().lower())
test_hi_norm = set(df_test_hi["text"].apply(norm))
test_en_norm = set(df_test_en["text"].apply(norm))

leakage_count = inc_curated["text"].apply(norm).isin(test_hi_norm | test_en_norm).sum()
print(f"Identified test set overlaps in curated incoming data: {leakage_count}")

# Purge leaks
inc_clean = inc_curated[~inc_curated["text"].apply(norm).isin(test_hi_norm | test_en_norm)].copy()
print(f"Clean, non-overlapping curated rows: {len(inc_clean)}")

# Combine into augmented training dataset
df_train_augmented = pd.concat([
    df_train_en[["text", "sentiment"]],
    df_train_hi[["text", "sentiment"]],
    inc_clean[["text", "sentiment"]]
], ignore_index=True)

augmented_csv = os.path.join(DATASETS_DIR, "train_augmented_sentiment.csv")
df_train_augmented.to_csv(augmented_csv, index=False)
print(f"Saved augmented dataset: {augmented_csv} ({len(df_train_augmented)} rows)")
print(f"Class distribution:\n{df_train_augmented['sentiment'].value_counts().to_dict()}")

# Step 3: NLP Preprocessing
prep = NLPPreprocessor()
print("\nPreprocessing texts with full 10-stage NLP pipeline...")
t0 = time.time()
X_train_clean = [prep.preprocess_to_text(t) for t in df_train_augmented["text"].fillna("")]
y_train = df_train_augmented["sentiment"]

X_test_en_clean = [prep.preprocess_to_text(t) for t in df_test_en["text"].fillna("")]
y_test_en = df_test_en["sentiment"]

X_test_hi_clean = [prep.preprocess_to_text(t) for t in df_test_hi["text"].fillna("")]
y_test_hi = df_test_hi["sentiment"]
print(f"Preprocessing completed in {time.time()-t0:.2f}s")

# Step 4: Multi-scale FeatureUnion TF-IDF
vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
    ("char", TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb", max_features=20000, sublinear_tf=True, min_df=5)),
], transformer_weights={"word": 1.5, "char": 0.5})

print("\nFitting vectorizer...")
t0 = time.time()
X_tr_vec = vectorizer.fit_transform(X_train_clean)
X_te_en_vec = vectorizer.transform(X_test_en_clean)
X_te_hi_vec = vectorizer.transform(X_test_hi_clean)
print(f"Vectorizer fitted in {time.time()-t0:.2f}s. Vocabulary size: {X_tr_vec.shape[1]}")

# Step 5: Fit Champion Logistic Regression
print("\nTraining Champion Logistic Regression...")
clf = LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42)
t0 = time.time()
clf.fit(X_tr_vec, y_train)
print(f"Model fitted in {time.time()-t0:.2f}s")

# Step 6: Evaluation
# English
t0 = time.time()
y_pred_en = clf.predict(X_te_en_vec)
lat_en = ((time.time() - t0) / len(y_test_en)) * 1000

acc_en = float(accuracy_score(y_test_en, y_pred_en))
prec_macro_en = float(precision_score(y_test_en, y_pred_en, average="macro"))
rec_macro_en = float(recall_score(y_test_en, y_pred_en, average="macro"))
f1_macro_en = float(f1_score(y_test_en, y_pred_en, average="macro"))
f1_weighted_en = float(f1_score(y_test_en, y_pred_en, average="weighted"))
cm_en = confusion_matrix(y_test_en, y_pred_en, labels=["negative", "neutral", "positive"]).tolist()

# Hinglish
t0 = time.time()
y_pred_hi = clf.predict(X_te_hi_vec)
lat_hi = ((time.time() - t0) / len(y_test_hi)) * 1000

acc_hi = float(accuracy_score(y_test_hi, y_pred_hi))
prec_macro_hi = float(precision_score(y_test_hi, y_pred_hi, average="macro"))
rec_macro_hi = float(recall_score(y_test_hi, y_pred_hi, average="macro"))
f1_macro_hi = float(f1_score(y_test_hi, y_pred_hi, average="macro"))
f1_weighted_hi = float(f1_score(y_test_hi, y_pred_hi, average="weighted"))
cm_hi = confusion_matrix(y_test_hi, y_pred_hi, labels=["negative", "neutral", "positive"]).tolist()

print("\n" + "="*70)
print(f"EVALUATION METRICS:")
print(f"English Holdout Set  (N=2,881): Acc={acc_en*100:.2f}%, Macro F1={f1_macro_en:.4f}, Weighted F1={f1_weighted_en:.4f}, Latency={lat_en:.3f}ms")
print(f"Hinglish Holdout Set (N=120)  : Acc={acc_hi*100:.2f}%, Macro F1={f1_macro_hi:.4f}, Weighted F1={f1_weighted_hi:.4f}, Latency={lat_hi:.3f}ms")
print("="*70)

# Class-level report for English
report_dict = classification_report(y_test_en, y_pred_en, output_dict=True)
class_metrics = {}
for c in ["negative", "neutral", "positive"]:
    class_metrics[c] = {
        "precision": float(report_dict[c]["precision"]),
        "recall": float(report_dict[c]["recall"]),
        "f1_score": float(report_dict[c]["f1-score"]),
        "support": float(report_dict[c]["support"]),
    }

# Step 7: Serialize metadata & artifacts
metadata = {
    "model_name": "Multinomial Logistic Regression (Bilingual L2, Balanced)",
    "algorithm": "logistic_regression",
    "classes": ["negative", "neutral", "positive"],
    "training_dataset_size": 11524,
    "bilingual_dataset_size": len(df_train_augmented),
    "testing_dataset_size": len(df_test_en),
    "hinglish_testing_dataset_size": len(df_test_hi),
    "vocabulary_size": int(X_tr_vec.shape[1]),
    "ngram_range": [1, 3],
    "sublinear_tf": True,
    "max_features": 65000,
    "accuracy": acc_en,
    "precision_macro": prec_macro_en,
    "recall_macro": rec_macro_en,
    "f1_macro": f1_macro_en,
    "f1_weighted": f1_weighted_en,
    "hinglish_accuracy": acc_hi,
    "hinglish_precision_macro": prec_macro_hi,
    "hinglish_recall_macro": rec_macro_hi,
    "hinglish_f1_macro": f1_macro_hi,
    "hinglish_f1_weighted": f1_weighted_hi,
    "class_metrics": class_metrics,
    "confusion_matrix": cm_en,
    "hinglish_confusion_matrix": cm_hi,
    "latency_ms_per_sample": lat_en,
    "hyperparameters": {
        "C": 1.25,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    },
    "training_date": datetime.now(timezone.utc).isoformat()
}

model_file = os.path.join(SAVED_DIR, "model.joblib")
vec_file = os.path.join(SAVED_DIR, "vectorizer.joblib")
meta_file = os.path.join(SAVED_DIR, "metadata.json")

joblib.dump(clf, model_file, compress=3)
joblib.dump(vectorizer, vec_file, compress=3)
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nArtifacts successfully written:")
print(f"  Model:      {model_file} ({os.path.getsize(model_file)/1024:.1f} KB)")
print(f"  Vectorizer: {vec_file} ({os.path.getsize(vec_file)/1024:.1f} KB)")
print(f"  Metadata:   {meta_file}")
