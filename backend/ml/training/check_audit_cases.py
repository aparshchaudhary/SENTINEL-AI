import os, sys, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

BASE_DIR = r"C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI"
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from ml.preprocessing import NLPPreprocessor

DATASETS_DIR = os.path.join(BASE_DIR, "backend", "ml", "datasets")
df_train_en = pd.read_csv(os.path.join(DATASETS_DIR, "train_sentiment.csv"))
df_train_hi = pd.read_csv(os.path.join(DATASETS_DIR, "train_hinglish_sentiment.csv"))
inc_df = pd.read_csv(os.path.join(DATASETS_DIR, "new_sentiment_incoming.tsv"), sep="\t")
inc_curated = inc_df[inc_df["id"].str.slice(4).astype(int) <= 270].copy()
test_hi = pd.read_csv(os.path.join(DATASETS_DIR, "test_hinglish_sentiment.csv"))

import re
def norm(t): return re.sub(r"\s+", " ", str(t).strip().lower())
test_hi_norm = set(test_hi["text"].apply(norm))
inc_curated = inc_curated[~inc_curated["text"].apply(norm).isin(test_hi_norm)]

df_tr = pd.concat([df_train_en[["text", "sentiment"]], df_train_hi[["text", "sentiment"]], inc_curated[["text", "sentiment"]]], ignore_index=True)
prep = NLPPreprocessor()
X_tr = [prep.preprocess_to_text(t) for t in df_tr["text"]]
y_tr = df_tr["sentiment"]

vec = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
    ("char", TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb", max_features=20000, sublinear_tf=True, min_df=5)),
], transformer_weights={"word": 1.5, "char": 0.5})

X_tr_vec = vec.fit_transform(X_tr)
clf = LogisticRegression(C=1.25, class_weight="balanced", max_iter=1000, random_state=42)
clf.fit(X_tr_vec, y_tr)

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

classes = list(clf.classes_)
for text, exp, lang in failure_cases_audit:
    feat = vec.transform([prep.preprocess_to_text(text)])
    pred = clf.predict(feat)[0]
    prob = clf.predict_proba(feat)[0]
    p_dict = {cls: round(float(prob[classes.index(cls)]), 3) for cls in classes}
    ok = (pred == exp)
    print(f"[{'PASS' if ok else 'FAIL'}] [{lang}] \"{text[:45]}\" -> Pred: {pred} | Exp: {exp} | Probs: {p_dict}")
