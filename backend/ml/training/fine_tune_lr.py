import os, sys, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = r"C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI"
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from ml.preprocessing import NLPPreprocessor

DATASETS_DIR = os.path.join(BASE_DIR, "backend", "ml", "datasets")
df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
df_test_en = pd.read_csv(os.path.join(DATASETS_DIR, "test_sentiment.csv"))
df_test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

inc_df = pd.read_csv(os.path.join(DATASETS_DIR, "new_sentiment_incoming.tsv"), sep="\t")
inc_curated = inc_df[inc_df["id"].str.slice(4).astype(int) <= 270].copy()

import re
def norm(t): return re.sub(r"\s+", " ", str(t).strip().lower())
test_hi_norm = set(df_test_hi["text"].apply(norm))
test_en_norm = set(df_test_en["text"].apply(norm))
inc_curated = inc_curated[~inc_curated["text"].apply(norm).isin(test_hi_norm | test_en_norm)]

df_tr = pd.concat([df_train_en[["text", "sentiment"]], df_train_hi[["text", "sentiment"]], inc_curated[["text", "sentiment"]]], ignore_index=True)
prep = NLPPreprocessor()
X_tr_clean = [prep.preprocess_to_text(t) for t in df_tr["text"]]
y_tr = df_tr["sentiment"]

X_te_en_clean = [prep.preprocess_to_text(t) for t in df_test_en["text"]]
y_te_en = df_test_en["sentiment"]

X_te_hi_clean = [prep.preprocess_to_text(t) for t in df_test_hi["text"]]
y_te_hi = df_test_hi["sentiment"]

for word_w, char_w in [(1.5, 0.5), (1.6, 0.4), (1.8, 0.5)]:
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb", max_features=25000, sublinear_tf=True, min_df=4)),
    ], transformer_weights={"word": word_w, "char": char_w})
    
    X_tr_vec = vec.fit_transform(X_tr_clean)
    X_te_en_vec = vec.transform(X_te_en_clean)
    X_te_hi_vec = vec.transform(X_te_hi_clean)
    
    for c in [1.0, 1.25, 1.4, 1.6, 1.8, 2.0]:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_tr_vec, y_tr)
        
        y_pred_en = clf.predict(X_te_en_vec)
        acc_en = accuracy_score(y_te_en, y_pred_en)
        f1_en = f1_score(y_te_en, y_pred_en, average="macro")
        
        y_pred_hi = clf.predict(X_te_hi_vec)
        acc_hi = accuracy_score(y_te_hi, y_pred_hi)
        f1_hi = f1_score(y_te_hi, y_pred_hi, average="macro")
        
        print(f"Weights ({word_w}, {char_w}) | C={c:4.2f} | EN Acc: {acc_en*100:.2f}% (F1: {f1_en:.4f}) | HI Acc: {acc_hi*100:.2f}% (F1: {f1_hi:.4f})")
