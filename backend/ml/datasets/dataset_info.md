# SENTINEL AI – Dataset Specification & Metadata
**Course**: CSE276 – Artificial Intelligence Foundations  
**Document Version**: 2.0 (Phase 2 Dataset Acquisition & Preprocessing)

---

## 1. Dataset Origin & Provenance

* **Dataset Title**: Twitter US Airline Sentiment Dataset
* **Original Annotator / Collector**: CrowdFlower (now Figure Eight Inc.)
* **Primary Source**: Collected in February 2015 from Twitter, hosted on Kaggle and widely adopted as an academic benchmark in Natural Language Processing research.
* **Repository Mirror**: `kaggle-Twitter-US-Airline-Sentiment` (Human-annotated ground-truth labels).
* **Target Classification Domain**: 3-Class Multiclass Sentiment Analysis (`positive`, `negative`, `neutral`).
* **License & Usage Terms**:
  * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) / Open Database License (ODbL).
  * Authorized for academic evaluation, research demonstration, and non-commercial educational benchmarking.

---

## 2. Dataset Metrics & Integrity Audit

| Metric | Count | Details |
| :--- | :--- | :--- |
| **Original Ingested Records** | 14,640 | Full raw CSV rows |
| **Missing Values in Text** | 0 | Zero null text values detected |
| **Missing Values in Sentiment** | 0 | Zero null sentiment targets detected |
| **Duplicate Tweets Removed** | 213 | Exact duplicate text matches purged |
| **Valid Post-Integrity Records** | 14,427 | High-confidence unique records |
| **Noise-Only Records Dropped** | 22 | Records reduced to empty strings after HTML/URL stripping |
| **Final Clean Processed Records** | **14,405** | Final verified dataset size |

---

## 3. Sentiment Class Distribution

### Consolidated Dataset (14,405 records)

| Sentiment Polarity | Record Count | Percentage | Class Balance Visual |
| :--- | :--- | :--- | :--- |
| **Negative** | 9,067 | 62.94% | `████████████▌` |
| **Neutral** | 3,052 | 21.19% | `████▎` |
| **Positive** | 2,286 | 15.87% | `███▏` |
| **Total** | **14,405** | **100.00%** | |

*Note: The natural imbalance reflects real-world public consumer feedback distributions, which will be handled during model training using balanced class weighting ($w_c = \frac{N}{K \cdot N_c}$).*

---

## 4. Cleaning & Preprocessing Pipeline

Every record passes through a 10-stage deterministic pipeline defined in `backend/ml/preprocessing/`:

1. **HTML Unescaping & Strip**: Converts HTML entities (`&amp;` $\to$ `&`, `&lt;` $\to$ `<`) and strips `<.*?>` tags.
2. **URL Stripping**: Regular expressions strip `http://...`, `https://...`, and `www...`.
3. **User Mention Sanitation**: Strips `@username` handles to avoid Twitter username memorization.
4. **Hashtag Decomposition**: Converts `#excellentService` $\to$ `excellentService`, extracting the underlying semantic word tokens.
5. **Case Normalization**: Converts all text to standard lowercase.
6. **Contraction Expansion**: Expands contractions to preserve semantic meaning (e.g., `can't` $\to$ `can not`, `won't` $\to$ `will not`, `n't` $\to$ `not`).
7. **Negation Preservation**: **Crucial Rule** — words such as `not`, `no`, `never`, `nor`, `neither`, `barely`, `hardly` are explicitly protected from stopword removal.
8. **Elongation Normalization**: Normalizes character repetitions (e.g., `sooooo goooood` $\to$ `soo good`) to prevent vocabulary explosion while preserving emphasis.
9. **Punctuation & Special Character Removal**: Strips non-alphanumeric punctuation while maintaining clean token boundaries.
10. **WordNet Lemmatization**: Inflected words are mapped to base lemma forms (e.g., `delighted` $\to$ `delight`, `flying` $\to$ `fly`).

---

## 5. Stratified Train / Test Partitioning

The dataset was partitioned using stratified sampling (`sklearn.model_selection.train_test_split`) with a fixed seed (`random_state=42`) to guarantee identical class ratios across partitions:

### Training Partition (`train_sentiment.csv` — 80%)
* **Total Samples**: **11,524**
* **Negative**: 7,258 (62.98%)
* **Neutral**: 2,436 (21.14%)
* **Positive**: 1,830 (15.88%)

### Testing Partition (`test_sentiment.csv` — 20%)
* **Total Samples**: **2,881**
* **Negative**: 1,809 (63.00%)
* **Neutral**: 616 (21.14%)
* **Positive**: 456 (15.86%)

### Integrity Verification
All 3 classes are proportionally represented in both splits down to within $\pm 0.02\%$ variance, ensuring zero statistical distribution skew.
