"""
SENTINEL AI – Dataset Validation, Cleaning & Preparation Pipeline
Academic Project: CSE276 – Artificial Intelligence Foundations

Validates, cleans, normalizes labels, applies NLP preprocessing,
and generates stratified 80/20 train/test splits.
"""

import os
import sys
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Add backend directory to path for clean imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing import NLPPreprocessor


class DatasetPreparer:
    """
    Orchestrates dataset loading, validation, NLP preprocessing,
    and stratified train-test splitting.
    """

    ALLOWED_LABELS = {"positive", "negative", "neutral"}

    def __init__(
        self,
        raw_dataset_path: str,
        output_dir: str,
        text_col: str = "text",
        label_col: str = "airline_sentiment",
        random_state: int = 42,
    ):
        self.raw_dataset_path = raw_dataset_path
        self.output_dir = output_dir
        self.text_col = text_col
        self.label_col = label_col
        self.random_state = random_state
        self.preprocessor = NLPPreprocessor()

    def validate_and_load(self) -> pd.DataFrame:
        """
        Loads the dataset and executes comprehensive validation.
        """
        print(f"[1/5] Loading raw dataset from: {self.raw_dataset_path}")
        if not os.path.exists(self.raw_dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.raw_dataset_path}")

        df = pd.read_csv(self.raw_dataset_path)
        original_count = len(df)
        print(f"      Initial record count: {original_count:,}")

        # Check required columns
        for col in [self.text_col, self.label_col]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from dataset. Available: {df.columns.tolist()}")

        # Keep only required columns to reduce memory overhead
        df = df[[self.text_col, self.label_col]].copy()
        df = df.rename(columns={self.text_col: "text", self.label_col: "sentiment"})

        # Missing value validation
        missing_text = df["text"].isna().sum()
        missing_label = df["sentiment"].isna().sum()
        print(f"      Missing values check - Text: {missing_text}, Sentiment: {missing_label}")
        df = df.dropna(subset=["text", "sentiment"])

        # Label normalization
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
        invalid_labels = set(df["sentiment"].unique()) - self.ALLOWED_LABELS
        if invalid_labels:
            raise ValueError(f"Found invalid sentiment labels: {invalid_labels}. Allowed: {self.ALLOWED_LABELS}")

        # Empty string validation
        df = df[df["text"].str.strip() != ""]

        # Duplicate detection and removal
        initial_records = len(df)
        df = df.drop_duplicates(subset=["text"])
        duplicates_removed = initial_records - len(df)
        print(f"      Duplicate records removed: {duplicates_removed}")

        print(f"      Valid records after integrity verification: {len(df):,}")
        print("      Class distribution after validation:")
        for label, count in df["sentiment"].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"        - {label.upper():8s}: {count:,} ({pct:.2f}%)")

        return df

    def apply_nlp_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning, tokenization, stopword removal, and lemmatization.
        """
        print(f"\n[2/5] Executing NLP Preprocessing on {len(df):,} records...")

        # Apply deterministic preprocessing
        df["cleaned_text"] = df["text"].apply(self.preprocessor.preprocess_to_text)

        # Remove rows where cleaned text became empty (e.g. only URLs or punctuation)
        empty_cleaned = (df["cleaned_text"].str.strip() == "").sum()
        if empty_cleaned > 0:
            print(f"      Dropping {empty_cleaned} records where cleaned text was empty.")
            df = df[df["cleaned_text"].str.strip() != ""].copy()

        print(f"      Cleaned dataset total records: {len(df):,}")
        return df

    def split_and_save(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs stratified 80/20 train/test splitting and writes outputs.
        """
        print(f"\n[3/5] Performing Stratified 80/20 Train/Test Split (random_state={self.random_state})...")

        X = df[["text", "cleaned_text"]]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=self.random_state,
            stratify=y,
        )

        train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

        print(f"      Training set size: {len(train_df):,} records (80%)")
        print(f"      Testing set size:  {len(test_df):,} records (20%)")

        # Verify class representation in both sets
        print("\n[4/5] Class representation verification:")
        print("      Training set distribution:")
        for label, count in train_df["sentiment"].value_counts().items():
            pct = (count / len(train_df)) * 100
            print(f"        - {label.upper():8s}: {count:,} ({pct:.2f}%)")

        print("      Testing set distribution:")
        for label, count in test_df["sentiment"].value_counts().items():
            pct = (count / len(test_df)) * 100
            print(f"        - {label.upper():8s}: {count:,} ({pct:.2f}%)")

        # Save outputs
        print("\n[5/5] Exporting prepared datasets to CSV...")
        os.makedirs(self.output_dir, exist_ok=True)

        train_path = os.path.join(self.output_dir, "train_sentiment.csv")
        test_path = os.path.join(self.output_dir, "test_sentiment.csv")
        full_path = os.path.join(self.output_dir, "processed_sentiment.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        df.to_csv(full_path, index=False)

        print(f"      Saved: {train_path}")
        print(f"      Saved: {test_path}")
        print(f"      Saved: {full_path}")

        return train_df, test_df


def main():
    datasets_dir = os.path.join(BASE_DIR, "ml", "datasets")
    raw_path = os.path.join(datasets_dir, "raw_sentiment_data.csv")

    preparer = DatasetPreparer(
        raw_dataset_path=raw_path,
        output_dir=datasets_dir,
        text_col="text",
        label_col="airline_sentiment",
        random_state=42,
    )

    df_valid = preparer.validate_and_load()
    df_clean = preparer.apply_nlp_preprocessing(df_valid)
    train_df, test_df = preparer.split_and_save(df_clean)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"Total Valid Processed Records: {len(df_clean):,}")
    print(f"Train Split: {len(train_df):,} | Test Split: {len(test_df):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
