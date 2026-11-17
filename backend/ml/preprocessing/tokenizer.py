"""
SENTINEL AI – NLP Preprocessing & Tokenization Module
Academic Project: CSE276 – Artificial Intelligence Foundations

Implements tokenization, negation-preserving stopword filtering,
and WordNet lemmatization for sentiment analysis.
"""

import os
from typing import List, Optional, Set

# Enable NLTK downloads through egress proxies on cloud environments
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .text_cleaner import TextCleaner


class NLPPreprocessor:
    """
    Modular, deterministic NLP preprocessor.
    Combines cleaning, tokenization, sentiment-aware stopword removal,
    and lemmatization into a reusable pipeline.
    """

    # Explicit negation words that MUST be preserved for accurate sentiment classification
    NEGATION_WORDS: Set[str] = {
        "not",
        "no",
        "never",
        "nor",
        "neither",
        "barely",
        "hardly",
        "scarcely",
        "none",
        "without",
    }

    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        use_lemmatization: bool = True,
        preserve_negations: bool = True,
        min_token_length: int = 2,
    ):
        """
        Initializes the NLP Preprocessor.

        Args:
            cleaner: TextCleaner instance (creates default if None).
            use_lemmatization: Whether to apply WordNet lemmatization.
            preserve_negations: Whether to protect negation words from stopword filtering.
            min_token_length: Minimum character length for retained tokens.
        """
        self.cleaner = cleaner or TextCleaner()
        self.use_lemmatization = use_lemmatization
        self.preserve_negations = preserve_negations
        self.min_token_length = min_token_length

        # Initialize WordNet lemmatizer
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

        # Build sentiment-safe stopword vocabulary
        base_stopwords = set(stopwords.words("english"))
        if self.preserve_negations:
            # Remove all negation markers from the stopwords set
            self.stop_words = base_stopwords - self.NEGATION_WORDS
        else:
            self.stop_words = base_stopwords

    def tokenize(self, text: Optional[str]) -> List[str]:
        """
        Cleans, tokenizes, filters stopwords, and lemmatizes the text.

        Args:
            text: Raw input string.

        Returns:
            List of preprocessed token strings.
        """
        # 1. Clean and normalize input
        cleaned_text = self.cleaner.clean(text)
        if not cleaned_text:
            return []

        # 2. Tokenize using NLTK word_tokenize
        tokens = word_tokenize(cleaned_text)

        processed_tokens: List[str] = []
        for token in tokens:
            # 3. Filter by token length (except single-letter negations if any)
            if len(token) < self.min_token_length and token not in self.NEGATION_WORDS:
                continue

            # 4. Filter stopwords while strictly preserving negations
            if token in self.stop_words:
                continue

            # 5. WordNet Lemmatization (nouns and verbs)
            if self.use_lemmatization and self.lemmatizer:
                # Lemmatize as verb first, then noun if unchanged
                lemma = self.lemmatizer.lemmatize(token, pos=wordnet.VERB)
                if lemma == token:
                    lemma = self.lemmatizer.lemmatize(token, pos=wordnet.NOUN)
                token = lemma

            processed_tokens.append(token)

        return processed_tokens

    def preprocess_to_text(self, text: Optional[str]) -> str:
        """
        End-to-end preprocessing pipeline returning a space-joined cleaned string.
        Specifically designed as input for scikit-learn's TfidfVectorizer.

        Args:
            text: Raw input string.

        Returns:
            Preprocessed string ready for vectorization.
        """
        tokens = self.tokenize(text)
        return " ".join(tokens)
