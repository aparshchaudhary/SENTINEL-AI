"""
SENTINEL AI – Text Cleaning Module
Academic Project: CSE276 – Artificial Intelligence Foundations

Provides deterministic text sanitation, normalization, regex-based cleaning,
and negation-preserving transformations.
"""

import html
import re
from typing import Optional


class TextCleaner:
    """
    A robust, reusable text cleaner for sentiment analysis.
    Preserves semantic polarity cues while eliminating noise, HTML, URLs,
    and unnecessary characters.
    """

    # Comprehensive contraction mapping preserving critical negation tokens
    CONTRACTIONS = {
        r"\bcan['’]t\b": "can not",
        r"\bwon['’]t\b": "will not",
        r"\bshan['’]t\b": "shall not",
        r"\bn['’]t\b": " not",
        r"\blet['’]s\b": "let us",
        r"\bthere['’]s\b": "there is",
        r"\bthat['’]s\b": "that is",
        r"\bwhat['’]s\b": "what is",
        r"\bhere['’]s\b": "here is",
        r"\bhow['’]s\b": "how is",
        r"\bit['’]s\b": "it is",
        r"\bi['’]m\b": "i am",
        r"\bhe['’]s\b": "he is",
        r"\bshe['’]s\b": "she is",
        r"\bwe['’]re\b": "we are",
        r"\bthey['’]re\b": "they are",
        r"\byou['’]re\b": "you are",
        r"\bi['’]ve\b": "i have",
        r"\byou['’]ve\b": "you have",
        r"\bwe['’]ve\b": "we have",
        r"\bthey['’]ve\b": "they have",
        r"\bi['’]ll\b": "i will",
        r"\byou['’]ll\b": "you will",
        r"\bhe['’]ll\b": "he will",
        r"\bshe['’]ll\b": "she will",
        r"\bwe['’]ll\b": "we will",
        r"\bthey['’]ll\b": "they will",
        r"\bi['’]d\b": "i would",
        r"\byou['’]d\b": "you would",
        r"\bhe['’]d\b": "he would",
        r"\bshe['’]d\b": "she would",
        r"\bwe['’]d\b": "we would",
        r"\bthey['’]d\b": "they would",
    }

    # Precompiled regex patterns for maximum execution efficiency
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    HTML_PATTERN = re.compile(r"<.*?>")
    USER_MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")
    NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-zA-Z\s]")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, strip_mentions: bool = True):
        self.strip_mentions = strip_mentions

    def clean(self, text: Optional[str]) -> str:
        """
        Cleans and normalizes the input text.

        Args:
            text: Raw input string.

        Returns:
            Normalized, clean string suitable for tokenization.
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        # 1. Unescape HTML entities (e.g., &amp; -> &, &quot; -> ")
        text = html.unescape(text)

        # 2. Strip HTML tags
        text = self.HTML_PATTERN.sub(" ", text)

        # 3. Strip URLs
        text = self.URL_PATTERN.sub(" ", text)

        # 4. Handle user mentions (@user)
        if self.strip_mentions:
            text = self.USER_MENTION_PATTERN.sub(" ", text)

        # 5. Extract words from hashtags (e.g., #awesome -> awesome)
        text = self.HASHTAG_PATTERN.sub(r"\1", text)

        # 6. Lowercase conversion for case normalization
        text = text.lower()

        # 7. Expand contractions (preserves critical negation tokens)
        for pattern, replacement in self.CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)

        # 8. Normalize elongated repeated characters (e.g., "sooooo" -> "soo", "greatttt" -> "greatt")
        # Collapsing to 2 letters preserves expressive intent while regularizing vocabulary
        text = self.REPEATED_CHARS_PATTERN.sub(r"\1\1", text)

        # 9. Remove non-alphabetic / non-whitespace characters (punctuation, special symbols)
        text = self.NON_ALPHANUMERIC_PATTERN.sub(" ", text)

        # 10. Normalize consecutive whitespaces and strip boundaries
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text
