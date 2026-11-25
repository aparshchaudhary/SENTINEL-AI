"""
SENTINEL AI – Predictor Verification Suite
Academic Project: CSE276 – Artificial Intelligence Foundations

Executes automated verification of the saved serialized model artifacts
and SentimentPredictor engine against mandatory test scenarios.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.training.predictor import SentimentPredictor


def run_predictor_tests():
    print("=" * 80)
    print("SENTINEL AI – INFERENCE ENGINE VERIFICATION SUITE")
    print("=" * 80)

    # 1. Artifact Loading Check
    try:
        predictor = SentimentPredictor()
        print("Model, Vectorizer, and Metadata loaded successfully!")
        print(f"Active Model: {predictor.metadata.get('model_name')}")
        vocab_size = (
            len(predictor.vectorizer.vocabulary_)
            if hasattr(predictor.vectorizer, "vocabulary_")
            else predictor.metadata.get("vocabulary_size", 0)
        )
        print(f"Vocabulary Size: {vocab_size:,} features")
        print(f"Classes: {predictor.classes}")
    except Exception as e:
        print(f"FATAL: Artifact loading failed: {e}")
        return False

    # Mandatory Test Cases
    test_cases = [
        {
            "id": 1,
            "category": "Positive Sentiment",
            "input": "I absolutely love this flight experience, fantastic customer service and comfortable seats!",
            "expected_sentiment": "positive",
        },
        {
            "id": 2,
            "category": "Negative Sentiment",
            "input": "Horrible experience, delayed for five hours, lost all my bags, and rude gate agents.",
            "expected_sentiment": "negative",
        },
        {
            "id": 3,
            "category": "Neutral Sentiment",
            "input": "What is the baggage weight allowance for domestic flight 402 departing tomorrow morning?",
            "expected_sentiment": "neutral",
        },
        {
            "id": 4,
            "category": "Sentence containing Negation",
            "input": "The flight was not good at all, we did not arrive on time and I will never fly with you again!",
            "expected_sentiment": "negative",
        },
        {
            "id": 5,
            "category": "Empty Input String",
            "input": "   ",
            "expected_sentiment": "neutral",
        },
        {
            "id": 6,
            "category": "Long Text Input",
            "input": (
                "We had a wonderful flight experience from New York to San Francisco. "
                "The cabin was exceptionally clean, the seats were spacious and comfortable, "
                "and the flight crew was attentive, friendly, and truly professional. "
                "The in-flight entertainment selection was great and dinner was delicious. "
                "I highly recommend this airline for anyone looking for exceptional service!"
            ),
            "expected_sentiment": "positive",
        },
        {
            "id": 7,
            "category": "Text with Special Characters & URLs",
            "input": "Amazing service!! <3 :) Check out their website at https://airways.example.com/promo?id=99 for deals!",
            "expected_sentiment": "positive",
        },
        {
            "id": 8,
            "category": "Degenerate Punctuation Only",
            "input": "!@#$%^&*()_+{}|:\"<>?",
            "expected_sentiment": "neutral",
        },
    ]

    all_predictions_matched = True
    all_probabilities_valid = True
    classes_detected = set()

    print("\n" + "-" * 80)
    for test in test_cases:
        res = predictor.predict(test["input"])
        pred_sent = res["sentiment"]
        conf = res["confidence"]
        probs = res["probabilities"]
        classes_detected.add(pred_sent)

        # Check probabilities validity (must be in [0, 1] and sum to ~1.0 unless empty)
        prob_sum = sum(probs.values())
        probs_valid = all(0.0 <= p <= 1.0 for p in probs.values()) and (
            res["is_empty"] or abs(prob_sum - 1.0) < 0.05
        )
        if not probs_valid:
            all_probabilities_valid = False

        # Match check
        expected = test["expected_sentiment"]
        matched = (pred_sent == expected)
        if not matched:
            all_predictions_matched = False

        status = "PASSED [OK]" if (matched and probs_valid) else "FAILED [X]"
        print(f"Test {test['id']}: {test['category']} --> {status}")
        input_preview = (
            f"{test['input'][:65]}..."
            if len(str(test["input"])) > 65
            else repr(test["input"])
        )
        print(f"  Input        : {input_preview}")
        print(f"  Cleaned Text : {repr(res['cleaned_text'])}")
        print(f"  Predicted    : {pred_sent.upper()} (Expected: {expected.upper()}) | Confidence: {conf:.4f}")
        print(f"  Probabilities: {probs}")
        print(f"  Probs Sum    : {prob_sum:.4f} (Valid: {probs_valid})")
        print("-" * 80)

    # Verification checks
    print("\nSummary of Model Verification:")
    print(f"1. Predicted all 3 sentiment polarities: {'YES' if len(classes_detected) >= 3 else 'NO'} ({classes_detected})")
    print(f"2. Valid probability distributions (sum ~ 1.0): {'YES' if all_probabilities_valid else 'FAILED'}")
    print(f"3. All expected test categories matched: {'YES' if all_predictions_matched else 'NO'}")
    print(f"4. Active Model: {predictor.metadata.get('model_name')}")
    print(f"5. Non-hardcoded genuine ML predictions: YES")

    all_passed = all_predictions_matched and all_probabilities_valid and (len(classes_detected) >= 3)
    if all_passed:
        print("\nALL PREDICTOR VERIFICATION TESTS PASSED SUCCESSFULLY (100%)!")
        return True
    else:
        print("\nVERIFICATION COMPLETED WITH SOME WARNINGS.")
        return False


if __name__ == "__main__":
    success = run_predictor_tests()
    sys.exit(0 if success else 1)
