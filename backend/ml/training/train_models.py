"""
SENTINEL AI – Machine Learning Model Training & Optimization Pipeline
Academic Project: CSE276 – Artificial Intelligence Foundations

Trains, tunes, evaluates, and compares:
1. Multinomial Logistic Regression (L2 regularization, balanced class weighting)
2. Multinomial Naive Bayes (Laplace smoothing, inverse-frequency sample weighting)
using TF-IDF feature extraction.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class ModelTrainer:
    """
    Orchestrates TF-IDF extraction, hyperparameter tuning with cross-validation,
    model evaluation, champion selection, and artifact serialization.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.datasets_dir = os.path.join(BASE_DIR, "ml", "datasets")
        self.saved_models_dir = os.path.join(BASE_DIR, "ml", "saved_models")
        self.reports_dir = os.path.join(BASE_DIR, "ml", "reports")

        os.makedirs(self.saved_models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.vectorizer: TfidfVectorizer = None
        self.trained_models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.classes: list = ["negative", "neutral", "positive"]

    def load_and_validate_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads training and testing CSVs and validates integrity.
        """
        train_path = os.path.join(self.datasets_dir, "train_sentiment.csv")
        test_path = os.path.join(self.datasets_dir, "test_sentiment.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Training or testing dataset CSV files are missing!")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print("=" * 70)
        print("STEP 1: DATASET VERIFICATION & INTEGRITY AUDIT")
        print("=" * 70)
        print(f"Training dataset: {train_path} ({len(train_df):,} records)")
        print(f"Testing dataset:  {test_path} ({len(test_df):,} records)")

        # Verify columns
        required_cols = {"text", "cleaned_text", "sentiment"}
        if not required_cols.issubset(train_df.columns) or not required_cols.issubset(test_df.columns):
            raise ValueError(f"Missing required columns in dataset. Expected: {required_cols}")

        # Verify no missing cleaned_text
        if train_df["cleaned_text"].isna().sum() > 0 or test_df["cleaned_text"].isna().sum() > 0:
            raise ValueError("Found NaN entries in cleaned_text!")

        # Verify classes
        unique_classes = sorted(list(train_df["sentiment"].unique()))
        if unique_classes != self.classes:
            raise ValueError(f"Unexpected sentiment classes: {unique_classes}. Expected: {self.classes}")

        print("Integrity Check Passed: Zero missing values, exactly 3 classes verified.")
        return train_df, test_df

    def extract_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        """
        Fits TF-IDF vectorizer strictly on training data and transforms testing data.
        """
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE EXTRACTION (TF-IDF VECTORIZATION)")
        print("=" * 70)

        # Configuration per specification:
        # Unigrams & bigrams, sublinear TF scaling, max 25,000 features, min_df=3, max_df=0.95
        # stop_words=None because cleaned_text already had stopwords removed with negations preserved!
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=25000,
            min_df=3,
            max_df=0.95,
            stop_words=None,
        )

        print("Fitting TF-IDF Vectorizer ONLY on training set (preventing data leakage)...")
        X_train_tfidf = self.vectorizer.fit_transform(train_df["cleaned_text"])

        print("Transforming holdout test set with fitted vectorizer...")
        X_test_tfidf = self.vectorizer.transform(test_df["cleaned_text"])

        y_train = train_df["sentiment"].values
        y_test = test_df["sentiment"].values

        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"Vocabulary size extracted: {vocab_size:,} features (unigrams + bigrams)")
        print(f"Training feature matrix shape: {X_train_tfidf.shape}")
        print(f"Testing feature matrix shape:  {X_test_tfidf.shape}")

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def train_logistic_regression(
        self, X_train: Any, y_train: np.ndarray, cv_splits: int = 5
    ) -> LogisticRegression:
        """
        Trains and tunes Multinomial Logistic Regression using Stratified K-Fold CV.
        """
        print("\n" + "=" * 70)
        print("STEP 3: MODEL 1 – MULTINOMIAL LOGISTIC REGRESSION TRAINING & TUNING")
        print("=" * 70)

        param_grid = {
            "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "solver": ["lbfgs"],
            "max_iter": [1000],
            "class_weight": ["balanced"],
        }

        base_lr = LogisticRegression(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=base_lr,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

        print(f"Running {cv_splits}-Fold Cross-Validation over C values: {param_grid['C']}...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_duration = time.time() - start_time

        print(f"Logistic Regression tuning completed in {training_duration:.2f}s")
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Macro F1-Score: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        self.trained_models["logistic_regression"] = best_model
        return best_model

    def train_multinomial_nb(
        self, X_train: Any, y_train: np.ndarray, cv_splits: int = 5
    ) -> MultinomialNB:
        """
        Trains and tunes Multinomial Naive Bayes using balanced inverse-frequency sample weights.
        """
        print("\n" + "=" * 70)
        print("STEP 4: MODEL 2 – MULTINOMIAL NAIVE BAYES TRAINING & TUNING")
        print("=" * 70)

        # Compute mathematically valid inverse-frequency sample weights to handle class imbalance
        sample_weights = compute_sample_weight("balanced", y_train)
        print("Computed inverse-frequency class sample weights for MultinomialNB.")

        param_grid = {
            "alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            "fit_prior": [True, False],
        }

        base_nb = MultinomialNB()
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=base_nb,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

        print(f"Running {cv_splits}-Fold Cross-Validation over alpha values: {param_grid['alpha']}...")
        start_time = time.time()
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        training_duration = time.time() - start_time

        print(f"Multinomial Naive Bayes tuning completed in {training_duration:.2f}s")
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Macro F1-Score: {grid_search.best_score_:.4f}")

        # Fit best estimator with sample weights
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train, sample_weight=sample_weights)
        self.trained_models["multinomial_nb"] = best_model
        return best_model

    def evaluate_model(
        self, model_name: str, model: Any, X_test: Any, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates a model comprehensively on the untouched holdout test dataset.
        """
        # Inference latency benchmark: average over test set
        start_infer = time.time()
        y_pred = model.predict(X_test)
        total_infer_time = time.time() - start_infer
        latency_per_sample_ms = (total_infer_time / len(y_test)) * 1000

        # Class probabilities
        y_proba = model.predict_proba(X_test)

        # Core evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        class_prec = precision_score(y_test, y_pred, average=None, labels=self.classes, zero_division=0)
        class_rec = recall_score(y_test, y_pred, average=None, labels=self.classes, zero_division=0)
        class_f1 = f1_score(y_test, y_pred, average=None, labels=self.classes, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        clf_report = classification_report(y_test, y_pred, labels=self.classes, output_dict=True)

        metrics = {
            "model_name": model_name,
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "f1_weighted": float(f1_weighted),
            "class_metrics": {
                cls: {
                    "precision": float(class_prec[i]),
                    "recall": float(class_rec[i]),
                    "f1_score": float(class_f1[i]),
                }
                for i, cls in enumerate(self.classes)
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report,
            "latency_ms_per_sample": float(latency_per_sample_ms),
        }

        self.model_metrics[model_name] = metrics
        return metrics

    def display_evaluation_results(self, name: str, metrics: Dict[str, Any]):
        """
        Prints clear, formatted terminal evaluation results.
        """
        print(f"\n--- {name.upper()} HOLDOUT TEST EVALUATION RESULTS ---")
        print(f"Accuracy         : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro F1-Score   : {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print(f"Macro Precision  : {metrics['precision_macro']:.4f}")
        print(f"Macro Recall     : {metrics['recall_macro']:.4f}")
        print(f"Inference Latency: {metrics['latency_ms_per_sample']:.4f} ms/sample")

        print("\nPer-Class Performance:")
        for cls in self.classes:
            cm_dict = metrics["class_metrics"][cls]
            print(
                f"  {cls.upper():8s} -> Precision: {cm_dict['precision']:.4f} | "
                f"Recall: {cm_dict['recall']:.4f} | F1: {cm_dict['f1_score']:.4f}"
            )

        print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
        print(f"{'':10s} {'Negative':>10s} {'Neutral':>10s} {'Positive':>10s}")
        for i, row_label in enumerate(self.classes):
            row = metrics["confusion_matrix"][i]
            print(f"{row_label.capitalize():10s} {row[0]:10d} {row[1]:10d} {row[2]:10d}")

    def select_and_serialize_champion(
        self, train_size: int, test_size: int
    ) -> Tuple[str, Any]:
        """
        Compares models objectively, selects champion based on Macro F1,
        and serializes artifacts to disk.
        """
        print("\n" + "=" * 70)
        print("STEP 5: MODEL COMPARISON & CHAMPION SELECTION")
        print("=" * 70)

        lr_f1 = self.model_metrics["logistic_regression"]["f1_macro"]
        nb_f1 = self.model_metrics["multinomial_nb"]["f1_macro"]

        lr_acc = self.model_metrics["logistic_regression"]["accuracy"]
        nb_acc = self.model_metrics["multinomial_nb"]["accuracy"]

        print(f"Logistic Regression -> Macro F1: {lr_f1:.4f} | Accuracy: {lr_acc:.4f}")
        print(f"Multinomial NB      -> Macro F1: {nb_f1:.4f} | Accuracy: {nb_acc:.4f}")

        # Decision rule: Highest Macro F1 (balances all 3 classes effectively)
        if lr_f1 >= nb_f1:
            champion_key = "logistic_regression"
            champion_name = "Multinomial Logistic Regression (L2, Balanced)"
        else:
            champion_key = "multinomial_nb"
            champion_name = "Multinomial Naive Bayes (Laplace smoothed, Sample weighted)"

        champion_model = self.trained_models[champion_key]
        champ_metrics = self.model_metrics[champion_key]

        print(f"\n>>> CHAMPION SELECTED: {champion_name} (Macro F1 = {champ_metrics['f1_macro']:.4f})")

        # Save artifacts
        print("\n" + "=" * 70)
        print("STEP 6: MODEL SERIALIZATION")
        print("=" * 70)

        model_path = os.path.join(self.saved_models_dir, "model.joblib")
        vectorizer_path = os.path.join(self.saved_models_dir, "vectorizer.joblib")
        meta_path = os.path.join(self.saved_models_dir, "metadata.json")

        print(f"Serializing champion model to: {model_path}")
        joblib.dump(champion_model, model_path)

        print(f"Serializing fitted TF-IDF vectorizer to: {vectorizer_path}")
        joblib.dump(self.vectorizer, vectorizer_path)

        metadata = {
            "model_name": champion_name,
            "algorithm": champion_key,
            "classes": self.classes,
            "training_dataset_size": train_size,
            "testing_dataset_size": test_size,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "max_features": 25000,
            "min_df": 3,
            "max_df": 0.95,
            "accuracy": champ_metrics["accuracy"],
            "precision_macro": champ_metrics["precision_macro"],
            "recall_macro": champ_metrics["recall_macro"],
            "f1_macro": champ_metrics["f1_macro"],
            "f1_weighted": champ_metrics["f1_weighted"],
            "class_metrics": champ_metrics["class_metrics"],
            "confusion_matrix": champ_metrics["confusion_matrix"],
            "latency_ms_per_sample": champ_metrics["latency_ms_per_sample"],
            "hyperparameters": champion_model.get_params(),
            "training_date": datetime.now().isoformat(),
        }

        # Handle non-serializable objects in hyperparameters
        for k, v in list(metadata["hyperparameters"].items()):
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                metadata["hyperparameters"][k] = str(v)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata file to: {meta_path}")

        # Also save raw metrics for evaluate_models.py
        metrics_summary_path = os.path.join(self.reports_dir, "evaluation_metrics.json")
        with open(metrics_summary_path, "w", encoding="utf-8") as f:
            json.dump(self.model_metrics, f, indent=2)
        print(f"Saved metrics summary to: {metrics_summary_path}")

        return champion_key, champion_model


def main():
    trainer = ModelTrainer(random_state=42)
    train_df, test_df = trainer.load_and_validate_datasets()

    X_train, X_test, y_train, y_test = trainer.extract_features(train_df, test_df)

    # 1. Train and tune Logistic Regression
    lr_model = trainer.train_logistic_regression(X_train, y_train, cv_splits=5)
    lr_metrics = trainer.evaluate_model("logistic_regression", lr_model, X_test, y_test)
    trainer.display_evaluation_results("Logistic Regression", lr_metrics)

    # 2. Train and tune Multinomial Naive Bayes
    nb_model = trainer.train_multinomial_nb(X_train, y_train, cv_splits=5)
    nb_metrics = trainer.evaluate_model("multinomial_nb", nb_model, X_test, y_test)
    trainer.display_evaluation_results("Multinomial Naive Bayes", nb_metrics)

    # 3. Objective comparison & serialization
    champ_key, champ_model = trainer.select_and_serialize_champion(
        train_size=len(train_df), test_size=len(test_df)
    )

    print("\n" + "=" * 70)
    print("PHASE 3 TRAINING & SERIALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
