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

    # Emojis & Emoticons mapping to sentiment polarity tokens
    EMOJI_POS = [":)", ":-)", ":D", ":-D", ";)", ";-)", "<3", "😊", "😄", "😃", "😁", "❤️", "👍", "🔥", "✨", "🎉", "👌", "😍"]
    EMOJI_NEG = [":(", ":-(", ":'(", ":'-((", ":/", ":-/", "😡", "😠", "😢", "😭", "👎", "💔", "🤮", "💩", "😤", "🤬", "😒", "😞"]

    # Sentiment idioms, Hinglish normalizations, and compound phrase patterns
    COMPOUND_PHRASES = [
        # English idioms & negations
        (re.compile(r"\bnot\s+bad\b", re.IGNORECASE), "not_bad"),
        (re.compile(r"\bno\s+problem\b", re.IGNORECASE), "no_problem"),
        (re.compile(r"\bno\s+issues?\b", re.IGNORECASE), "no_problem"),
        (re.compile(r"\bnothing\s+more\s*,?\s*nothing\s+less\b", re.IGNORECASE), "nothing_more_nothing_less"),
        (re.compile(r"\bnothing\s+special\b", re.IGNORECASE), "nothing_special"),
        (re.compile(r"\bincredible\s+disaster\b", re.IGNORECASE), "incredible_disaster"),
        (re.compile(r"\bcomplete\s+disaster\b", re.IGNORECASE), "complete_disaster"),
        (re.compile(r"\btotal\s+disaster\b", re.IGNORECASE), "total_disaster"),
        (re.compile(r"\bhuge\s+disaster\b", re.IGNORECASE), "huge_disaster"),
        (re.compile(r"\bnot\s+good\b", re.IGNORECASE), "not_good"),
        (re.compile(r"\bnot\s+happy\b", re.IGNORECASE), "not_happy"),

        # Hinglish spelling normalizations
        (re.compile(r"\baccha\b", re.IGNORECASE), "achha"),
        (re.compile(r"\bachhaa\b", re.IGNORECASE), "achha"),
        (re.compile(r"\bachhi\b", re.IGNORECASE), "achhi"),
        (re.compile(r"\bbadiya\b", re.IGNORECASE), "badhiya"),
        (re.compile(r"\bbadhia\b", re.IGNORECASE), "badhiya"),
        (re.compile(r"\bthik\b", re.IGNORECASE), "theek"),
        (re.compile(r"\bthek\b", re.IGNORECASE), "theek"),
        (re.compile(r"\bthik-thak\b", re.IGNORECASE), "theek_thaak"),
        (re.compile(r"\btheek-thaak\b", re.IGNORECASE), "theek_thaak"),
        (re.compile(r"\bbe-?kaar\b", re.IGNORECASE), "bekar"),
        (re.compile(r"\bbakwas\b", re.IGNORECASE), "bakwaas"),
        (re.compile(r"\bshandaar\b", re.IGNORECASE), "shandar"),
        (re.compile(r"\bzbrdst\b", re.IGNORECASE), "zabardast"),
        (re.compile(r"\bbht\b", re.IGNORECASE), "bahut"),
        (re.compile(r"\bbohot\b", re.IGNORECASE), "bahut"),
        (re.compile(r"\bbahot\b", re.IGNORECASE), "bahut"),
        (re.compile(r"\bnahin\b", re.IGNORECASE), "nahi"),
        (re.compile(r"\bnhi\b", re.IGNORECASE), "nahi"),
        (re.compile(r"\bpaise\s+waste\b", re.IGNORECASE), "paisa_waste"),
        (re.compile(r"\bpaisa\s+waste\b", re.IGNORECASE), "paisa_waste"),
        (re.compile(r"\bpaisa\s+vasool\b", re.IGNORECASE), "paisa_vasool"),

        # Hinglish compound negation phrases
        (re.compile(r"\bpasand\s+nahi\b", re.IGNORECASE), "nahi_pasand"),
        (re.compile(r"\bnahi\s+pasand\b", re.IGNORECASE), "nahi_pasand"),
        (re.compile(r"\bachha\s+nahi\b", re.IGNORECASE), "nahi_achha"),
        (re.compile(r"\bnahi\s+achha\b", re.IGNORECASE), "nahi_achha"),
        (re.compile(r"\bachhi\s+nahi\b", re.IGNORECASE), "nahi_achhi"),
        (re.compile(r"\bnahi\s+achhi\b", re.IGNORECASE), "nahi_achhi"),
        (re.compile(r"\btheek\s+nahi\b", re.IGNORECASE), "nahi_theek"),
        (re.compile(r"\bnahi\s+theek\b", re.IGNORECASE), "nahi_theek"),
        (re.compile(r"\bkuch\s+khaas\s+nahi\b", re.IGNORECASE), "kuch_khaas_nahi"),
        (re.compile(r"\bkuchh\s+khaas\s+nahi\b", re.IGNORECASE), "kuch_khaas_nahi"),
        (re.compile(r"\bproblem\s+nahi\b", re.IGNORECASE), "koi_problem_nahi"),
        (re.compile(r"\bkoi\s+problem\s+nahi\b", re.IGNORECASE), "koi_problem_nahi"),
        (re.compile(r"\bkoi\s+dikkat\s+nahi\b", re.IGNORECASE), "koi_dikkat_nahi"),
        (re.compile(r"\bkoi\s+issue\s+nahi\b", re.IGNORECASE), "koi_problem_nahi"),
        (re.compile(r"\bbilkul\s+bekar\b", re.IGNORECASE), "bilkul_bekar"),
        (re.compile(r"\bbilkul\s+nahi\b", re.IGNORECASE), "bilkul_nahi"),
    ]

    # Precompiled regex patterns for maximum execution efficiency
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    HTML_PATTERN = re.compile(r"<.*?>")
    USER_MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")
    
    # Preserves Latin (a-zA-Z), Devanagari Hindi (\u0900-\u097F), underscores for compound tokens (_), and whitespace
    CLEAN_ALLOWED_PATTERN = re.compile(r"[^a-zA-Z\u0900-\u097F\s_]")
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

        # 6. Map sentiment emojis & emoticons before lowercasing
        for em in self.EMOJI_POS:
            text = text.replace(em, " __emoji_pos__ ")
        for em in self.EMOJI_NEG:
            text = text.replace(em, " __emoji_neg__ ")

        # 7. Lowercase conversion for case normalization
        text = text.lower()

        # 8. Expand contractions (preserves critical negation tokens)
        for pattern, replacement in self.CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)

        # 9. Normalize compound phrases and Hinglish spelling
        for pattern, replacement in self.COMPOUND_PHRASES:
            text = pattern.sub(replacement, text)

        # 10. Normalize elongated repeated characters (e.g., "sooooo" -> "soo", "greatttt" -> "greatt")
        text = self.REPEATED_CHARS_PATTERN.sub(r"\1\1", text)

        # 11. Remove non-alphabetic / non-Devanagari characters (preserves Devanagari Hindi and compound underscores)
        text = self.CLEAN_ALLOWED_PATTERN.sub(" ", text)

        # 12. Normalize consecutive whitespaces and strip boundaries
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text
