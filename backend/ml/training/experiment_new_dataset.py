"""
SENTINEL AI: Systematic Experimentation with New Ingested Dataset
Evaluates multiple dataset compositions, feature extractors, and classifiers.
Strict zero-data-leakage verification on English (2,881) and Hinglish (120) holdout sets.
"""

import os
import sys
import time
import json
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing import NLPPreprocessor

DATASETS_DIR = os.path.join(BASE_DIR, "ml", "datasets")

# 1. Load Existing Datasets
df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_test_en = pd.read_csv(os.path.join(DATASETS_DIR, "test_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
df_test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

# 2. Load Incoming Dataset
inc_df = pd.read_csv(os.path.join(DATASETS_DIR, "new_sentiment_incoming.tsv"), sep="\t")

# Clean incoming dataset
inc_df["sentiment"] = inc_df["sentiment"].str.strip().str.lower()
inc_df["language"] = inc_df["language"].replace({"En": "English"}).fillna("English")

# Normalization for leakage check
def norm(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())

test_en_norm = set(df_test_en["text"].apply(norm))
test_hi_norm = set(df_test_hi["text"].apply(norm))

# Identify leaks
leak_en = inc_df[inc_df["text"].apply(norm).isin(test_en_norm)]
leak_hi = inc_df[inc_df["text"].apply(norm).isin(test_hi_norm)]

print(f"Incoming rows: {len(inc_df)}")
print(f"Leakage with English test set: {len(leak_en)}")
print(f"Leakage with Hinglish test set: {len(leak_hi)}")
for _, r in leak_hi.iterrows():
    print(f"  [PURGED TEST LEAK] ID: {r['id']} | {r['sentiment']} | {r['text']}")

# Filter out all test leaks
valid_inc_df = inc_df[~inc_df["text"].apply(norm).isin(test_en_norm | test_hi_norm)].copy()
print(f"Valid non-overlapping incoming rows: {len(valid_inc_df)}")

# Curated seed vs synthetic
curated_df = valid_inc_df[valid_inc_df["id"].str.slice(4).astype(int) <= 270].copy()
synth_df = valid_inc_df[valid_inc_df["id"].str.slice(4).astype(int) > 270].copy()

print(f"Curated rows: {len(curated_df)} (Pos: {(curated_df['sentiment']=='positive').sum()}, Neg: {(curated_df['sentiment']=='negative').sum()}, Neu: {(curated_df['sentiment']=='neutral').sum()})")
print(f"Synthetic rows: {len(synth_df)} (Pos: {(synth_df['sentiment']=='positive').sum()}, Neg: {(synth_df['sentiment']=='negative').sum()}, Neu: {(synth_df['sentiment']=='neutral').sum()})")

# Prepare Preprocessor
prep = NLPPreprocessor()
print("\nPreprocessing test sets...")
X_test_en_clean = [prep.preprocess_to_text(t) for t in df_test_en["text"].fillna("")]
y_test_en = df_test_en["sentiment"]

X_test_hi_clean = [prep.preprocess_to_text(t) for t in df_test_hi["text"].fillna("")]
y_test_hi = df_test_hi["sentiment"]

# Define Dataset Compositions to benchmark
# Config 1: Baseline (11,524 EN + 528 HI = 12,052)
df_train_base = pd.concat([
    df_train_en[["text", "sentiment"]],
    df_train_hi[["text", "sentiment"]]
], ignore_index=True)

# Config 2: Full Ingestion (12,052 + 1,042 = 13,094)
df_train_full = pd.concat([
    df_train_en[["text", "sentiment"]],
    df_train_hi[["text", "sentiment"]],
    valid_inc_df[["text", "sentiment"]]
], ignore_index=True)

# Config 3: Curated Only (12,052 + 267 = 12,319) - high quality, balanced neutral
df_train_curated = pd.concat([
    df_train_en[["text", "sentiment"]],
    df_train_hi[["text", "sentiment"]],
    curated_df[["text", "sentiment"]]
], ignore_index=True)

# Config 4: Curated + Controlled De-duplicated Synthetic (12,052 + 267 + 100 sample = 12,419)
# To avoid class imbalance starvation of Neutral
synth_pos_sample = synth_df[synth_df["sentiment"] == "positive"].sample(n=60, random_state=42)
synth_neg_sample = synth_df[synth_df["sentiment"] == "negative"].sample(n=60, random_state=42)
df_train_balanced_synth = pd.concat([
    df_train_en[["text", "sentiment"]],
    df_train_hi[["text", "sentiment"]],
    curated_df[["text", "sentiment"]],
    synth_pos_sample[["text", "sentiment"]],
    synth_neg_sample[["text", "sentiment"]]
], ignore_index=True)

configs = [
    ("Config 1: Current Baseline (N=12,052)", df_train_base),
    ("Config 2: Full Ingestion (N=13,094)", df_train_full),
    ("Config 3: Curated Ingestion (N=12,319)", df_train_curated),
    ("Config 4: Curated + Balanced Synth (N=12,439)", df_train_balanced_synth),
]

failure_cases_audit = [
    ("Not bad at all! I actually enjoyed it.", "positive", "EN"),
    ("It works exactly as described, nothing more, nothing less.", "neutral", "EN"),
    ("Wow, what an incredible disaster!", "negative", "EN"),
    ("The delivery was late, but the product is fantastic.", "positive", "EN"),
    ("The food was delicious, although the staff was rude.", "neutral", "EN"),
    ("Product theek hai, nothing special.", "neutral", "HI"),
    ("Yaar, mujhe iska experience bahut pasand aaya.", "positive", "HI"),
    ("Mujhe ye product bilkul pasand nahi aaya.", "negative", "HI"),
    ("Mujhe ye movie bahut achhi lagi.", "positive", "HI"),
    ("Ye phone bilkul bekar hai.", "negative", "HI"),
    ("Service average thi, kuch khaas nahi.", "neutral", "HI"),
    ("Mujhe is product se koi problem nahi hai.", "positive", "HI"),
    ("Bhai kya zabardast quality hai!", "positive", "HI"),
    ("Paise waste ho gaye, totally disappointed.", "negative", "HI"),
    ("Experience achha tha lekin customer support bekar tha.", "negative", "HI"),
    ("The package was delivered yesterday at 10 AM.", "neutral", "REG"),
]

all_results = []

for cfg_name, df_tr in configs:
    print(f"\n{'#'*80}\nEvaluating Dataset: {cfg_name}\n{'#'*80}")
    print(f"Sentiment distribution:\n{df_tr['sentiment'].value_counts().to_dict()}")
    
    t0 = time.time()
    X_tr_clean = [prep.preprocess_to_text(t) for t in df_tr["text"].fillna("")]
    y_tr = df_tr["sentiment"]
    print(f"Preprocessed in {time.time()-t0:.2f}s")
    
    # We test two strong vectorizers:
    # V1: FeatureUnion(Word 1-3 45k + Char 3-4 20k) -> Champion vectorizer
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb", max_features=20000, sublinear_tf=True, min_df=5)),
    ], transformer_weights={"word": 1.5, "char": 0.5})
    
    X_tr_vec = vec.fit_transform(X_tr_clean)
    X_te_en_vec = vec.transform(X_test_en_clean)
    X_te_hi_vec = vec.transform(X_test_hi_clean)
    
    classifiers = [
        ("Multinomial LR (C=1.25, Balanced)", LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42)),
        ("Multinomial LR (C=1.5, Balanced)", LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42)),
        ("Calibrated LinearSVC (C=0.3, Balanced)", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", random_state=42, dual=False))),
        ("Soft Voting Ensemble (Calibrated SVC + LR)", VotingClassifier(
            estimators=[
                ("svc", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", random_state=42, dual=False))),
                ("lr", LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42))
            ],
            voting="soft",
            weights=[1.2, 1.0]
        ))
    ]
    
    for clf_name, clf in classifiers:
        t_start = time.time()
        clf.fit(X_tr_vec, y_tr)
        t_train = time.time() - t_start
        
        # Test English
        t_inf0 = time.time()
        y_pred_en = clf.predict(X_te_en_vec)
        lat_en = ((time.time() - t_inf0) / len(y_test_en)) * 1000
        
        acc_en = accuracy_score(y_test_en, y_pred_en)
        macro_f1_en = f1_score(y_test_en, y_pred_en, average="macro")
        weighted_f1_en = f1_score(y_test_en, y_pred_en, average="weighted")
        
        # Test Hinglish
        t_inf1 = time.time()
        y_pred_hi = clf.predict(X_te_hi_vec)
        lat_hi = ((time.time() - t_inf1) / len(y_test_hi)) * 1000
        
        acc_hi = accuracy_score(y_test_hi, y_pred_hi)
        macro_f1_hi = f1_score(y_test_hi, y_pred_hi, average="macro")
        weighted_f1_hi = f1_score(y_test_hi, y_pred_hi, average="weighted")
        
        # Audit cases
        case_correct = 0
        for text, expected, lang in failure_cases_audit:
            c_clean = prep.preprocess_to_text(text)
            feat = vec.transform([c_clean])
            pred = clf.predict(feat)[0]
            if pred == expected:
                case_correct += 1
                
        all_results.append({
            "dataset": cfg_name,
            "classifier": clf_name,
            "acc_en": acc_en,
            "macro_f1_en": macro_f1_en,
            "weighted_f1_en": weighted_f1_en,
            "acc_hi": acc_hi,
            "macro_f1_hi": macro_f1_hi,
            "weighted_f1_hi": weighted_f1_hi,
            "cases_passed": f"{case_correct}/{len(failure_cases_audit)}",
            "train_time_s": round(t_train, 2),
            "latency_ms": round(lat_en, 4)
        })
        
        print(f"[{clf_name}] EN: {acc_en*100:.2f}% (F1: {macro_f1_en:.4f}) | HI: {acc_hi*100:.2f}% (F1: {macro_f1_hi:.4f}) | Cases: {case_correct}/16")

print("\n" + "="*110)
print("BENCHMARK COMPARISON MATRIX ACROSS ALL CONFIGURATIONS")
print("="*110)
res_df = pd.DataFrame(all_results)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(res_df.to_string(index=False))
