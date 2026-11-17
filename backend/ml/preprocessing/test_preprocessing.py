"""
SENTINEL AI – Preprocessing Pipeline Verification Suite
Academic Project: CSE276 – Artificial Intelligence Foundations

Validates preprocessing behavior against mandatory test scenarios:
1. Positive sentence
2. Negative sentence
3. Neutral sentence
4. Sentence containing negation
5. Sentence containing URLs
6. Sentence containing HTML
7. Empty / None input
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing import TextCleaner, NLPPreprocessor


def run_tests():
    preprocessor = NLPPreprocessor()
    cleaner = TextCleaner()

    test_cases = [
        {
            "id": 1,
            "category": "Positive Sentence",
            "input": "The flight was absolutely fantastic, the flight crew was remarkably polite and hospitable! #lovedIt",
            "check": lambda res: "fantastic" in res and "polite" in res and "lovedit" in res,
        },
        {
            "id": 2,
            "category": "Negative Sentence",
            "input": "Terrible experience, my luggage was completely lost and customer service was horribly rude.",
            "check": lambda res: "terrible" in res and "luggage" in res and "rude" in res,
        },
        {
            "id": 3,
            "category": "Neutral Sentence",
            "input": "Flight 1042 departed Denver at 08:30 AM and is scheduled to land in Chicago at 12:15 PM.",
            "check": lambda res: "flight" in res and "denver" in res and "chicago" in res,
        },
        {
            "id": 4,
            "category": "Sentence with Negation (CRITICAL)",
            "input": "This is NOT good at all, I did not receive my ticket and they never apologized!",
            "check": lambda res: "not" in res and "never" in res and "good" in res,
        },
        {
            "id": 5,
            "category": "Sentence with URLs",
            "input": "Check flight delay status updates at https://airline.com/status?id=1024 and www.tracking.org/flight",
            "check": lambda res: "http" not in res and "www" not in res and "delay" in res,
        },
        {
            "id": 6,
            "category": "Sentence with HTML & Entities",
            "input": "<div class='feedback'>Service was fast &amp; comfortable! <br/><p>Highly recommended.</p></div>",
            "check": lambda res: "<" not in res and ">" not in res and "fast" in res and "comfortable" in res,
        },
        {
            "id": 7,
            "category": "Empty and Whitespace Inputs",
            "input": "     \n\t   ",
            "check": lambda res: res == "",
        },
        {
            "id": 8,
            "category": "None (Null) Input",
            "input": None,
            "check": lambda res: res == "",
        },
    ]

    print("=" * 80)
    print("SENTINEL AI – NLP PREPROCESSING VALIDATION TEST SUITE")
    print("=" * 80)

    all_passed = True

    for test in test_cases:
        raw_text = test["input"]
        cleaned = cleaner.clean(raw_text)
        tokens = preprocessor.tokenize(raw_text)
        processed_str = preprocessor.preprocess_to_text(raw_text)
        passed = test["check"](processed_str)

        if not passed:
            all_passed = False

        status = "PASSED [OK]" if passed else "FAILED [X]"
        print(f"\nTest {test['id']}: {test['category']} --> {status}")
        print(f"  Raw Input     : {repr(raw_text)}")
        print(f"  Cleaned Text  : {repr(cleaned)}")
        print(f"  Token List    : {tokens}")
        print(f"  Final Output  : {repr(processed_str)}")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 8 PREPROCESSING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("TEST FAILURES DETECTED! PLEASE REVIEW LOGS ABOVE.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
