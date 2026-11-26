"""
Comprehensive Bilingual & Hinglish Model Training & Evaluation Suite
Compares:
1. Feature representations (Word TF-IDF, Char-wb TF-IDF, Multi-scale FeatureUnion)
2. Classifiers (LogisticRegression, CalibratedLinearSVC, VotingClassifier, SGDClassifier)
3. Evaluates on:
   - English Holdout Test Set (2,881 samples)
   - Hinglish Holdout Test Set (120 samples)
   - Specific Failure Cases from the audit
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing import TextCleaner, NLPPreprocessor

DATASETS_DIR = os.path.join(BASE_DIR, "ml", "datasets")

# Load Datasets
df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_test_en = pd.read_csv(os.path.join(DATASETS_DIR, "test_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
df_test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

# Build combined training set
# For df_train_en, keep text and sentiment
df_train_combined = pd.concat([
    df_train_en[["text", "sentiment"]],
    df_train_hi[["text", "sentiment"]]
], ignore_index=True)

print(f"Combined Training Set Size: {len(df_train_combined)} (EN: {len(df_train_en)}, HI: {len(df_train_hi)})")
print(f"English Holdout Test Set Size: {len(df_test_en)}")
print(f"Hinglish Holdout Test Set Size: {len(df_test_hi)}")

# Preprocessing
cleaner = TextCleaner()
preprocessor = NLPPreprocessor()

print("\nPreprocessing training and test texts...")
t0 = time.time()
X_train_raw = df_train_combined["text"].fillna("")
y_train = df_train_combined["sentiment"]

X_test_en_raw = df_test_en["text"].fillna("")
y_test_en = df_test_en["sentiment"]

X_test_hi_raw = df_test_hi["text"].fillna("")
y_test_hi = df_test_hi["sentiment"]

X_train_clean = [preprocessor.preprocess_to_text(t) for t in X_train_raw]
X_test_en_clean = [preprocessor.preprocess_to_text(t) for t in X_test_en_raw]
X_test_hi_clean = [preprocessor.preprocess_to_text(t) for t in X_test_hi_raw]
print(f"Preprocessing completed in {time.time()-t0:.2f}s")

# Define Feature Extractors
vec_word_45k = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=45000,
    sublinear_tf=True,
    min_df=2
)

vec_union_multiscale = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
    ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=35000, sublinear_tf=True)),
])

candidate_setups = [
    (
        "Candidate A: Multinomial LR (Word 1-2, 45k, C=1.25, Balanced)",
        vec_word_45k,
        LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42)
    ),
    (
        "Candidate B: Multinomial LR (Word+Char Union 80k, C=1.5, Balanced)",
        vec_union_multiscale,
        LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42)
    ),
    (
        "Candidate C: Calibrated LinearSVC (Word+Char Union 80k, C=0.3, Balanced)",
        vec_union_multiscale,
        CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", random_state=42, dual=False))
    ),
    (
        "Candidate D: Soft Voting Ensemble (Calibrated LinearSVC + LR, Balanced)",
        vec_union_multiscale,
        VotingClassifier(
            estimators=[
                ("svc", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", random_state=42, dual=False))),
                ("lr", LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42))
            ],
            voting="soft",
            weights=[1.5, 1.0]
        )
    ),
]

failure_cases_audit = [
    # English failure cases
    ("Not bad at all! I actually enjoyed it.", "positive", "EN"),
    ("It works exactly as described, nothing more, nothing less.", "neutral", "EN"),
    ("Wow, what an incredible disaster!", "negative", "EN"),
    ("The delivery was late, but the product is fantastic.", "positive", "EN"),
    ("The food was delicious, although the staff was rude.", "neutral", "EN"),
    # Hinglish failure cases
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
    # Mandatory regression check from test suite
    ("The package was delivered yesterday at 10 AM.", "neutral", "REG"),
]

results = []

for name, vec, clf in candidate_setups:
    print(f"\n{'='*75}\nEvaluating: {name}\n{'='*75}")
    t_start = time.time()
    
    # 1. Fit Vectorizer on Training data only
    X_tr_vec = vec.fit_transform(X_train_clean)
    X_te_en_vec = vec.transform(X_test_en_clean)
    X_te_hi_vec = vec.transform(X_test_hi_clean)
    
    # 2. Fit Classifier
    clf.fit(X_tr_vec, y_train)
    t_train = time.time() - t_start
    
    # 3. Evaluate English Holdout Test Set
    t_inf0 = time.time()
    y_pred_en = clf.predict(X_te_en_vec)
    inf_latency_en = ((time.time() - t_inf0) / len(y_test_en)) * 1000  # ms per sample
    
    acc_en = accuracy_score(y_test_en, y_pred_en)
    macro_f1_en = f1_score(y_test_en, y_pred_en, average="macro")
    weighted_f1_en = f1_score(y_test_en, y_pred_en, average="weighted")
    
    # 4. Evaluate Hinglish Holdout Test Set
    t_inf1 = time.time()
    y_pred_hi = clf.predict(X_te_hi_vec)
    inf_latency_hi = ((time.time() - t_inf1) / len(y_test_hi)) * 1000
    
    acc_hi = accuracy_score(y_test_hi, y_pred_hi)
    macro_f1_hi = f1_score(y_test_hi, y_pred_hi, average="macro")
    weighted_f1_hi = f1_score(y_test_hi, y_pred_hi, average="weighted")
    
    print(f"Training Time: {t_train:.2f}s")
    print(f"English Test Set  (N={len(y_test_en)}): Accuracy = {acc_en*100:.2f}%, Macro F1 = {macro_f1_en:.4f}, Weighted F1 = {weighted_f1_en:.4f}, Latency = {inf_latency_en:.2f}ms")
    print(f"Hinglish Test Set (N={len(y_test_hi)}): Accuracy = {acc_hi*100:.2f}%, Macro F1 = {macro_f1_hi:.4f}, Weighted F1 = {weighted_f1_hi:.4f}, Latency = {inf_latency_hi:.2f}ms")
    
    # 5. Test Audit Cases
    print("\n--- Failure Cases Audit Predictions ---")
    case_correct = 0
    classes = list(clf.classes_)
    
    for text, expected, lang in failure_cases_audit:
        c_clean = preprocessor.preprocess_to_text(text)
        feat = vec.transform([c_clean])
        pred = clf.predict(feat)[0]
        prob = clf.predict_proba(feat)[0] if hasattr(clf, "predict_proba") else [0, 0, 0]
        p_dict = {cls: round(float(prob[classes.index(cls)]), 3) for cls in classes}
        
        ok = (pred == expected)
        if ok:
            case_correct += 1
        status_str = "PASS" if ok else "FAIL"
        print(f"[{status_str}] [{lang}] '{text[:45]}...' -> Pred: {pred:8s} | Expected: {expected:8s} | Probs: {p_dict}")
        
    print(f"\nAudit Cases Passed: {case_correct} / {len(failure_cases_audit)} ({case_correct/len(failure_cases_audit)*100:.1f}%)")
    
    results.append({
        "name": name,
        "acc_en": acc_en,
        "macro_f1_en": macro_f1_en,
        "weighted_f1_en": weighted_f1_en,
        "acc_hi": acc_hi,
        "macro_f1_hi": macro_f1_hi,
        "weighted_f1_hi": weighted_f1_hi,
        "case_correct": case_correct,
        "train_time": t_train,
        "latency_ms": inf_latency_en
    })

print("\n" + "="*80)
print("FINAL BENCHMARK SUMMARY COMPARISON TABLE")
print("="*80)
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
