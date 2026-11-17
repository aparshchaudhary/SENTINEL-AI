# SENTINEL AI — Machine Learning Pipeline Architecture
**Course**: CSE276 – Artificial Intelligence Foundations  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  
**Document Version**: 1.0 (Phase 1 Specification)

---

## 1. Machine Learning Problem Formulation

* **Task**: Multiclass Supervised Natural Language Classification.
* **Input Space**: Unstructured textual sequence $X = [x_1, x_2, \dots, x_N]$.
* **Target Space**: Categorical label $Y \in \{ \text{negative}, \text{neutral}, \text{positive} \}$.
* **Core Philosophy**: Zero mock predictions. Genuine statistical NLP using Scikit-learn and NLTK, yielding real probability distributions $P(Y = c \mid X)$.

```
+---------------------------------------------------------------------------------------+
|                                    ML PIPELINE                                        |
+---------------------------------------------------------------------------------------+
|  1. Dataset Ingestion  -->  2. Cleaning & Regex   -->  3. NLTK Token & Lemmatization  |
|  4. TF-IDF Extraction  -->  5. Model Training     -->  6. Statistical Evaluation      |
|  7. Model Selection    -->  8. Joblib Export      -->  9. Production Runtime Serving  |
+---------------------------------------------------------------------------------------+
```

---

## 2. End-to-End Pipeline Stages

### Stage 1: Dataset Loading
* **Datasets Considered**: Multi-domain academic benchmark corpus / Twitter Financial & Review Sentiment datasets containing balanced distribution across Negative, Neutral, and Positive polarities.
* **Format**: Structured `.csv` or `.parquet` loaded via Pandas DataFrame.
* **Validation**:
  * Null / empty string removal.
  * Class balance inspection (ensuring no class dominates excessively, or calculating class weights).
  * Train/Test stratification: $80\%$ Training set, $20\%$ Independent Test set with fixed `random_state=42` for exact reproducibility.

### Stage 2: Data Cleaning
* Regex-driven cleaning engine designed to strip non-semantic noise while preserving emotive cues:
  1. Strip HTML tags, escape characters (`&amp;`, `<br>`, etc.).
  2. Strip raw URLs (`http://...`, `https://...`).
  3. Normalize or strip user mentions (`@username`) and hashtag symbols while keeping hashtag words.
  4. Normalize accented unicode characters to ASCII.
  5. Contraction expansion (e.g., "don't" -> "do not", "can't" -> "cannot") to preserve negation signals vital for sentiment.

### Stage 3: Text Preprocessing & NLP
* Natural Language Processing operations implemented via NLTK:
  1. **Case Normalization**: Lowercasing text.
  2. **Tokenization**: Regex / Word tokenization into discrete lexical tokens.
  3. **Stopword Filtering**: Removal of generic English stopwords (`the`, `is`, `at`, `which`), carefully excluding negation terms (`not`, `no`, `never`, `nor`) so negative polarity remains intact.
  4. **Lemmatization**: WordNet Lemmatizer reducing inflected words to their morphological base form (e.g., "running", "ran" -> "run", "better" -> "good").

### Stage 4: Feature Extraction (TF-IDF Vectorization)
* Converts preprocessed token sequences into numerical vector representations using `TfidfVectorizer` from `scikit-learn`:
  * **TF-IDF Formula**:
    $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  * **N-gram Range**: $(1, 2)$ — extracts both unigrams and bigrams (e.g., "not good", "very happy") capturing vital localized context.
  * **Max Features**: $15,000$ to $25,000$ top informative features.
  * **Sublinear TF Scaling**: `sublinear_tf=True` (replaces $\text{TF}$ with $1 + \log(\text{TF})$) to prevent high-frequency words from disproportionately dominating.
  * **Minimum Document Frequency**: `min_df=3` (discards extremely rare typographical errors).

### Stage 5: Model Training
We benchmark two foundational Machine Learning algorithms:

1. **Multinomial Naive Bayes (MNB)**:
   * Probabilistic model based on Bayes' theorem:
     $$P(c \mid d) \propto P(c) \prod_{k=1}^{V} P(t_k \mid c)^{f_k}$$
   * With Laplace smoothing ($\alpha = 1.0$).
   * Highly efficient, minimal training overhead, robust baseline.

2. **Multinomial Logistic Regression (Softmax Regression)**:
   * Linear model optimizing multinomial cross-entropy loss with $L_2$ regularization:
     $$P(Y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
   * Hyperparameter tuning across regularization strength $C \in [0.1, 1.0, 5.0, 10.0]$ and solver options (`lbfgs`, `saga`).
   * Provides well-calibrated posterior probabilities for confidence scoring.

### Stage 6: Model Evaluation
Evaluated rigorously on the unseen test split ($20\%$ holdout) across multiple evaluation metrics:
* **Accuracy**: Overall fraction of correct classifications.
* **Per-Class Precision, Recall, F1-Score**: Identifying if any specific class (e.g., Neutral) suffers from low recall.
* **Macro F1-Score**: Unweighted mean of F1-scores across all three classes:
  $$\text{Macro F1} = \frac{\text{F1}_{\text{negative}} + \text{F1}_{\text{neutral}} + \text{F1}_{\text{positive}}}{3}$$
* **Confusion Matrix**: Full $3 \times 3$ error matrix highlighting classification misdirections.
* **ROC-AUC (One-vs-Rest)**: Multi-class area under the Receiver Operating Characteristic curve.

### Stage 7: Model Selection
* Candidate models are compared based on:
  1. Primary metric: **Macro F1-Score** on holdout test set.
  2. Calibration: Reliability of predicted class probabilities.
  3. Inference Latency: Latency in milliseconds per sample (must be $<25\text{ms}$ for real-time interactivity).
* The champion model is flagged as the production deployment artifact.

### Stage 8: Model Serialization & Packaging
* Serialized using `joblib` into `backend/ml/saved_models/`:
  * `vectorizer.joblib`: The fitted TF-IDF feature vocabulary and IDF weights.
  * `model.joblib`: The fitted classifier weights and parameters.
  * `metadata.json`: Timestamp, hyperparameters, test set performance metrics, vocabulary size, class mapping dictionary.

### Stage 9: Production Inference Serving
* During application startup (`lifespan` in FastAPI), the artifacts are loaded into singleton memory:
  ```python
  # Runtime pipeline:
  clean_text = preprocessor.clean(raw_text)
  features = vectorizer.transform([clean_text])
  predicted_class_idx = model.predict(features)[0]
  probabilities = model.predict_proba(features)[0]
  confidence = float(np.max(probabilities))
  ```
* Thread-safe, sub-millisecond scoring per text input.
