import os, sys
BASE_DIR = r"C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI\backend"
sys.path.insert(0, BASE_DIR)
from ml.training.predictor import SentimentPredictor

p = SentimentPredictor()
test_texts = [
    "The scheduled flight UA104 departs at 4:15 PM from terminal 3, gate B12.",
    "Product theek hai, nothing special.",
    "Bhai kya zabardast quality hai! Yaar, mujhe iska experience bahut pasand aaya.",
    "Terrible customer service. Our luggage was completely lost.",
    "Not bad at all! I actually enjoyed it.",
    "The delivery was late, but the product is fantastic."
]

deterministic = True
for t in test_texts:
    base_res = p.predict(t)
    for _ in range(5):
        repeat_res = p.predict(t)
        if repeat_res["sentiment"] != base_res["sentiment"] or repeat_res["probabilities"] != base_res["probabilities"]:
            deterministic = False
            print(f"Nondeterminism found on: {t}")

print("Determinism verification: PASSED (100% Identical Outputs)" if deterministic else "FAILED")
for t in test_texts:
    res = p.predict(t)
    print(f"'{t[:40]}...' -> {res['sentiment']} (conf={res['confidence']}) {res['probabilities']}")
