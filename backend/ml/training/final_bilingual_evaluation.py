import sys, os
sys.path.insert(0, r"C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI\backend")
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from ml.preprocessing import NLPPreprocessor

prep = NLPPreprocessor()
DATASETS_DIR = r"C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI\backend\ml\datasets"

df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
df_train = pd.concat([df_train_en[["text", "sentiment"]], df_train_hi[["text", "sentiment"]]], ignore_index=True)

df_test_en = pd.read_csv(os.path.join(DATASETS_DIR, "test_sentiment.csv"))
df_test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

print("Pre-cleaning text...")
X_train = [prep.preprocess_to_text(t) for t in df_train["text"].fillna("")]
y_train = df_train["sentiment"]

X_test_en = [prep.preprocess_to_text(t) for t in df_test_en["text"].fillna("")]
y_test_en = df_test_en["sentiment"]

X_test_hi = [prep.preprocess_to_text(t) for t in df_test_hi["text"].fillna("")]
y_test_hi = df_test_hi["sentiment"]

vec = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
    ("char", TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb", max_features=20000, sublinear_tf=True, min_df=5)),
], transformer_weights={"word": 1.5, "char": 0.5})

print("Fitting vectorizer...")
t0 = time.time()
X_tr = vec.fit_transform(X_train)
X_te_en = vec.transform(X_test_en)
X_te_hi = vec.transform(X_test_hi)
print(f"Vectorizer fit time: {time.time()-t0:.2f}s, feature shape: {X_tr.shape}")

print("Fitting Multinomial Logistic Regression...")
t1 = time.time()
clf = LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42)
clf.fit(X_tr, y_train)
train_time = time.time() - t1
print(f"Model fit time: {train_time:.2f}s")

# Evaluate English Holdout
t_inf0 = time.time()
y_pred_en = clf.predict(X_te_en)
latency_en = (time.time() - t_inf0) / len(y_test_en) * 1000

acc_en = accuracy_score(y_test_en, y_pred_en)
macro_p_en = precision_score(y_test_en, y_pred_en, average="macro")
macro_r_en = recall_score(y_test_en, y_pred_en, average="macro")
macro_f1_en = f1_score(y_test_en, y_pred_en, average="macro")
weighted_f1_en = f1_score(y_test_en, y_pred_en, average="weighted")

print("\n" + "="*70)
print("ENGLISH HOLDOUT TEST SET EVALUATION (N=2,881)")
print("="*70)
print(f"Accuracy:    {acc_en*100:.2f}%")
print(f"Macro F1:    {macro_f1_en:.4f}")
print(f"Weighted F1: {weighted_f1_en:.4f}")
print(f"Macro Prec:  {macro_p_en:.4f}")
print(f"Macro Recall:{macro_r_en:.4f}")
print(f"Inference Latency: {latency_en:.2f} ms/query")
print("\nClassification Report (English):")
print(classification_report(y_test_en, y_pred_en, digits=4))
print("Confusion Matrix (English) [negative, neutral, positive]:")
print(confusion_matrix(y_test_en, y_pred_en, labels=["negative", "neutral", "positive"]))

# Evaluate Hinglish Holdout
t_inf1 = time.time()
y_pred_hi = clf.predict(X_te_hi)
latency_hi = (time.time() - t_inf1) / len(y_test_hi) * 1000

acc_hi = accuracy_score(y_test_hi, y_pred_hi)
macro_p_hi = precision_score(y_test_hi, y_pred_hi, average="macro")
macro_r_hi = recall_score(y_test_hi, y_pred_hi, average="macro")
macro_f1_hi = f1_score(y_test_hi, y_pred_hi, average="macro")
weighted_f1_hi = f1_score(y_test_hi, y_pred_hi, average="weighted")

print("\n" + "="*70)
print("HINGLISH / HINDI HOLDOUT TEST SET EVALUATION (N=120)")
print("="*70)
print(f"Accuracy:    {acc_hi*100:.2f}%")
print(f"Macro F1:    {macro_f1_hi:.4f}")
print(f"Weighted F1: {weighted_f1_hi:.4f}")
print(f"Macro Prec:  {macro_p_hi:.4f}")
print(f"Macro Recall:{macro_r_hi:.4f}")
print(f"Inference Latency: {latency_hi:.2f} ms/query")
print("\nClassification Report (Hinglish):")
print(classification_report(y_test_hi, y_pred_hi, digits=4))
print("Confusion Matrix (Hinglish) [negative, neutral, positive]:")
print(confusion_matrix(y_test_hi, y_pred_hi, labels=["negative", "neutral", "positive"]))
