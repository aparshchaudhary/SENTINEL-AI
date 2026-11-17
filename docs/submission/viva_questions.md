# SENTINEL AI – Comprehensive Academic Viva & Oral Defense Preparation Guide

**Academic Course**: CSE276 – Artificial Intelligence Foundations  
**Document**: 45 In-Depth Viva Voce Technical Questions & Detailed Answers  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  

---

## Section 1: Sentiment Analysis & NLP Fundamentals

### Q1: What is Sentiment Analysis, and how does SENTINEL AI formulate it mathematically?
**Answer**: Sentiment Analysis (opinion mining) is the computational study of opinions, sentiments, and emotions expressed in text. SENTINEL AI formulates it as a multi-class supervised text classification problem: mapping an input document $d$ from vocabulary space $\mathcal{X}$ to a discrete category $y \in \mathcal{Y} = \{\text{negative}, \text{neutral}, \text{positive}\}$.

### Q2: Why is three-class sentiment analysis harder than binary (positive/negative) classification?
**Answer**: In binary classification, sentiment polarities are polar opposites. In three-class classification, the **neutral** class introduces substantial semantic ambiguity. Neutral text often shares common vocabulary with both positive and negative remarks (e.g., flight schedules, factual status updates, questions). Furthermore, human annotator agreement on neutral labels is historically lower than on strong polarities, leading to label noise.

### Q3: What is the difference between a lexicon-based sentiment approach (like VADER) and machine learning?
**Answer**: Lexicon-based methods use predefined dictionaries of words mapped to polarity scores and apply heuristics (e.g., intensifiers, all-caps). They do not learn from domain corpora and struggle with domain-specific idioms. Machine learning approaches (like SENTINEL AI) statistically learn discriminative weights directly from labeled domain data (TF-IDF features &rarr; Logistic Regression weights), allowing the model to adapt to airline-specific vernacular and context.

### Q4: What is tokenization, and which tokenizer does SENTINEL AI use?
**Answer**: Tokenization is the process of splitting continuous character streams into discrete linguistic units (tokens/words). SENTINEL AI uses NLTK's `word_tokenize`, which uses the Punkt rule-based tokenizer trained on English punctuation patterns to split contractions, periods, and symbols accurately.

### Q5: What is lemmatization, and how does it differ from stemming?
**Answer**: Both reduce word variants to a base form. **Stemming** (e.g., Porter Stemmer) is a heuristic, rule-based process that chops off suffixes (often creating non-words like `"fliez"` for `"flies"`). **Lemmatization** uses vocabulary and morphological analysis with a lexical database (WordNet in SENTINEL AI) to return the actual valid dictionary root word (lemma), e.g., `"delayed"`, `"delaying"`, and `"delays"` all reduce to the lemma `"delay"`.

### Q6: How does SENTINEL AI handle syntactic negation, and why is it critical?
**Answer**: Standard stopword filtering removes words like `"not"`, `"no"`, and `"never"`. If a user writes `"The flight was not good"`, standard filtering yields `"flight good"`, inverting negative sentiment into positive. SENTINEL AI implements a regex negation-binding pass: whenever a negation cue (`not`, `no`, `never`, `n't`, `nor`) is detected, it is bound to the subsequent predicate token (e.g., `"not_good"`, `"never_helpful"`). These bound bigrams are preserved through stopword filtering and lemmatization, allowing the classifier to learn strongly negative weights for negated predicates.

---

## Section 2: Feature Engineering & TF-IDF

### Q7: What is TF-IDF, and what is its mathematical formula?
**Answer**: Term Frequency-Inverse Document Frequency is a numerical statistic reflecting how important a word is to a document in a collection.
- $\text{TF}(t, d) = \text{count}(t, d)$ (or $1 + \log(\text{count})$ for sublinear scaling).
- $\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$.
- $\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$.

### Q8: What is sublinear term frequency scaling, and why did you enable `sublinear_tf=True`?
**Answer**: Without sublinear scaling, a document containing the word `"delay"` twenty times receives twenty times the weight of a document containing it once. In real text, a word occurring twenty times is rarely twenty times as important. Sublinear scaling replaces raw term count with $1 + \log(\text{count})$, dampening the impact of excessively repetitive words in long reviews.

### Q9: What n-gram range was used, and what advantage do bigrams provide?
**Answer**: SENTINEL AI uses an n-gram range of $(1, 2)$ (unigrams and bigrams). Unigrams capture isolated topical words (`"comfortable"`, `"refund"`), while bigrams capture paired contextual phrases (`"customer service"`, `"lost luggage"`, `"on time"`), providing crucial semantic disambiguation.

### Q10: Why did you set `min_df=3` and `max_df=0.95`?
**Answer**: 
- `min_df=3` excludes n-grams that appear in fewer than 3 documents, filtering out typos, rare usernames, and idiosyncratic noise, reducing model dimensionality.
- `max_df=0.95` excludes terms appearing in more than 95% of documents, filtering out corpus-wide boilerplate that carries zero discriminative power.
- These cutoffs reduced the potential vocabulary from over 35,000 sparse n-grams down to 7,235 dense, high-signal features.

---

## Section 3: Machine Learning Algorithms & Optimization

### Q11: Explain how Multinomial Logistic Regression works for multi-class classification.
**Answer**: Multinomial Logistic Regression models the posterior probabilities of the $K$ classes via linear functions of the input feature vector $\mathbf{x}$. It computes a linear score $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$ for each class $k$, and normalizes these scores into a valid probability distribution using the **Softmax function**:
$$P(Y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
The class with the highest probability is chosen as the predicted label: $\hat{y} = \arg\max_k P(Y=k \mid \mathbf{x})$.

### Q12: What solver was used in Logistic Regression, and how does it optimize parameters?
**Answer**: The **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) solver was used. It is a quasi-Newton optimization algorithm that approximates the inverse Hessian matrix using only recent gradient evaluations, requiring $O(m \cdot d)$ memory instead of $O(d^2)$. It converges significantly faster than standard gradient descent on high-dimensional, convex smooth loss surfaces like multinomial cross-entropy.

### Q13: What does the hyperparameter $C$ represent in Scikit-learn's LogisticRegression?
**Answer**: $C$ is the inverse of the regularization strength ($C = \frac{1}{\lambda}$). A smaller $C$ (e.g., $C=0.1$) imposes stronger L2 regularization, shrinking feature weights closer to zero to prevent overfitting. A larger $C$ (e.g., $C=10$) penalizes weights less, fitting the training data more aggressively. Through cross-validation, $C=0.5$ proved optimal for SENTINEL AI, providing ideal balance between bias and variance.

### Q14: How does Multinomial Naive Bayes differ from Logistic Regression?
**Answer**: 
- **Naive Bayes is a Generative model**: It models the joint probability $P(\mathbf{x}, y) = P(y) \prod_i P(x_i \mid y)$ under the "naive" conditional independence assumption that words occur independently given the class.
- **Logistic Regression is a Discriminative model**: It directly models the conditional probability $P(y \mid \mathbf{x})$ without assuming feature independence. Because n-gram features (e.g., `"customer"` and `"customer service"`) are naturally correlated, Logistic Regression achieved superior accuracy (76.43% vs 73.12%) by learning correlated weights.

### Q15: What is class imbalance, and how does `class_weight='balanced'` address it?
**Answer**: In our dataset, negative remarks comprise 63% of the samples, while positive remarks comprise only 16%. Without intervention, standard loss functions favor predicting the majority negative class to minimize overall error. Setting `class_weight='balanced'` automatically weights the loss for each sample inversely proportional to its class frequency:
$$w_k = \frac{N}{K \cdot N_k}$$
This penalizes misclassifications of minority positive and neutral classes proportionally more, preventing model collapse.

---

## Section 4: Model Evaluation & Performance Metrics

### Q16: Why is Accuracy alone an inadequate metric for this dataset?
**Answer**: In an imbalanced dataset where 63% of records are negative, a naive dummy model that predicts "negative" 100% of the time achieves 63% accuracy while completely failing to detect positive or neutral sentiments. Therefore, **Macro F1-Score** and **Class-Wise Precision/Recall** are essential to evaluate performance fairly across all classes.

### Q17: Define Precision, Recall, and F1-Score in the context of Sentiment Analysis.
**Answer**: 
- **Precision**: Of all samples the model predicted as Positive, what fraction were genuinely Positive? $\frac{TP}{TP + FP}$.
- **Recall**: Of all genuinely Positive samples in the dataset, what fraction did the model successfully find? $\frac{TP}{TP + FN}$.
- **F1-Score**: The harmonic mean of Precision and Recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.

### Q18: What is the difference between Macro F1-Score and Weighted F1-Score?
**Answer**:
- **Macro F1**: Calculates the F1-score independently for each class and takes their unweighted arithmetic average: $\frac{F1_{\text{neg}} + F1_{\text{neu}} + F1_{\text{pos}}}{3}$. It treats all three classes equally, heavily penalizing models that fail on minority classes. SENTINEL AI achieved **0.7138**.
- **Weighted F1**: Calculates the average of class F1-scores weighted by the support (number of true instances) of each class. SENTINEL AI achieved **0.7712**.

### Q19: Explain the Confusion Matrix of SENTINEL AI on the holdout test set.
**Answer**: Evaluated on 2,881 unseen samples:
- **Negative (Row 0, 1,815 actual)**: 1,467 correctly classified as Negative (80.8% recall); 257 confused as Neutral; 91 confused as Positive.
- **Neutral (Row 1, 609 actual)**: 404 correctly classified as Neutral (66.3% recall); 132 confused as Negative; 73 confused as Positive.
- **Positive (Row 2, 457 actual)**: 331 correctly classified as Positive (72.4% recall); 83 confused as Neutral; 43 confused as Negative.
The diagonal ($1467 + 404 + 331 = 2,202$) represents correctly classified samples ($2202 / 2881 = 76.43\%$).

### Q20: What was the holdout accuracy and latency of the champion model?
**Answer**: Holdout accuracy is **76.43%**. Average single-sample inference latency is **15 to 17 milliseconds** (including text preprocessing, vector transformation, and Softmax calculation).

---

## Section 5: Backend Architecture & FastAPI

### Q21: Why choose FastAPI over Flask or Django?
**Answer**: 
1. **Asynchronous Architecture**: Built on Starlette and ASGI, FastAPI handles asynchronous I/O natively, providing high concurrent request throughput.
2. **Pydantic Validation**: Automatic schema enforcement and data serialization based on Python type hints.
3. **Auto-Generated Documentation**: Generates interactive OpenAPI 3.1 and Swagger UI documentation (`/docs`) directly from code without third-party plugins.
4. **Speed**: Benchmarks demonstrate performance competitive with Node.js and Go for JSON APIs.

### Q22: What is the FastAPI lifespan context manager, and how is it used in SENTINEL AI?
**Answer**: The `@asynccontextmanager lifespan(app: FastAPI)` runs setup logic before the web server begins receiving traffic and teardown logic on shutdown. In SENTINEL AI, it initializes the SQLite database schema (`init_db()`) and loads the ML model artifacts (`SentimentPredictor`) once into RAM, eliminating per-request deserialization latency.

### Q23: How are exceptions handled in the backend?
**Answer**: Exception handling is centralized in `backend/main.py`:
- `RequestValidationError`: Caught and converted to clean HTTP 422 JSON detailing invalid fields.
- `StarletteHTTPException`: Returns structured HTTP status codes with human-readable error descriptions.
- Generic `Exception`: Traps unhandled errors, logs the full stacktrace to server logs, and returns a sanitized HTTP 500 error preventing internal file path or database driver leakage.

### Q24: What is CORS, and what vulnerability did you fix in Phase 7?
**Answer**: Cross-Origin Resource Sharing (CORS) is a browser security mechanism that restricts web applications running at one origin from requesting resources from another origin. In Phase 7, we discovered `CORS_ORIGINS` contained `"*"` while `allow_credentials=True` was active. According to W3C specifications, browsers reject responses with `Access-Control-Allow-Origin: *` when credentials are included. We removed `"*"` and implemented an explicit whitelist and loopback regex matching `localhost` and `127.0.0.1`.

### Q25: How does the backend prevent path traversal attacks during file downloads?
**Answer**: In `download_batch_csv`, a client passes a `job_id`. We resolve the file path using `os.path.abspath` and verify that `resolved_path.startswith(os.path.abspath(settings.UPLOADS_DIR))`. If a client supplies path traversal sequences (such as `../../etc/passwd`), the check fails and returns HTTP 400 Access Denied.

---

## Section 6: Frontend Architecture & React 18

### Q26: What is the advantage of using Vite over Create React App (CRA)?
**Answer**: Create React App relies on Webpack, which bundles the entire application before starting the dev server, resulting in slow cold starts. Vite leverages native browser ES Modules (ESM) and esbuild (written in Go) for pre-bundling, starting the development server in under 500 milliseconds and updating changes instantly via Hot Module Replacement (HMR).

### Q27: How does SENTINEL AI manage client-side state?
**Answer**: Client state is managed using React declarative hooks:
- `useState`: For local form inputs, loading flags, and active results.
- `useEffect` and `useCallback`: For fetching backend telemetry on mount and executing debounced searches.
- `useContext` (`ThemeContext`): For global dark/light theme switching with persistence in browser `localStorage`.
- Server data is fetched dynamically via Axios without redundant global state libraries (Redux), keeping the bundle lightweight.

### Q28: How is code-splitting configured in the frontend?
**Answer**: Code-splitting is configured in `vite.config.js` via `build.rollupOptions.output.manualChunks`. Dependencies are partitioned into separate vendor chunks:
- `vendor-react` (React, ReactDOM, React Router): 163.9 kB
- `vendor-charts` (Recharts, D3 helpers): 424.7 kB
- `vendor-ui` (Lucide React icons): 17.2 kB
- Application code: 126.2 kB
This ensures no bundle exceeds 500 kB and clients only download vendor updates when dependencies change.

### Q29: How does the frontend handle empty states and error boundaries?
**Answer**: Every visual component (`SentimentPieChart`, `TrendsChart`, `History`) implements an `EmptyState` fallback when database records are zero, rendering helpful guidance instead of crashing or showing broken axes. When the backend is unreachable, `Navbar` renders a red "API Offline" badge, and `Dashboard` displays a non-blocking retry banner.

---

## Section 7: Database & Persistence

### Q30: Why choose SQLite for SENTINEL AI, and how is it connected?
**Answer**: SQLite was chosen for zero-configuration, ACID-compliant embedded persistence. It requires no standalone database server, enabling portable evaluation and grading. It connects via SQLAlchemy 2.0 ORM using `check_same_thread=False` and connection pooling in [session.py](file:///C:/Users/hp/.gemini/antigravity/scratch/SENTINEL-AI/backend/app/database/session.py).

### Q31: How are database transactions protected against failure?
**Answer**: All write operations (`db.add()`, `db.delete()`, `db.commit()`) are encapsulated in `try...except` blocks with explicit `db.rollback()` calls on error. This guarantees that if a constraint fails or a network timeout occurs, the database transaction is safely aborted, preventing session lockups.

### Q32: How is the database schema migrated or initialized?
**Answer**: Database tables are declared as SQLAlchemy declarative models inheriting from `Base`. The `init_db()` function executes `Base.metadata.create_all(bind=engine)` during application lifespan startup, automatically creating tables (`sentiment_analyses`, `batch_jobs`, `model_metrics`) if they do not exist.

---

## Section 8: Model Inference & Production Reliability

### Q33: How are model artifacts serialized and deserialized?
**Answer**: Artifacts are serialized using `joblib.dump()` to preserve Scikit-learn vocabulary mappings, coefficients, and IDF vectors. On startup, `joblib.load()` loads `model.joblib` and `vectorizer.joblib` into memory. A companion `metadata.json` stores evaluation benchmarks, hyperparameters, and class labels.

### Q34: What happens if an uploaded CSV file contains missing or null text values?
**Answer**: In `sentiment_service.py`, each row value is checked via `pd.notna(val)`. Null or missing cells are safely coerced to empty strings `""`, preventing pandas `NaN` exceptions from breaking the batch inference loop.

### Q35: How does the batch processor detect the text column in uploaded CSVs?
**Answer**: The system applies a two-tier strategy:
1. If the user provides an explicit `text_column` name, it validates that the column exists in the CSV (raising HTTP 400 if missing).
2. If omitted, an auto-detection heuristic checks for common column names (`text`, `cleaned_text`, `tweet`, `review`, `comment`, `feedback`, `content`).
3. If no match is found, it falls back to the first object/string column in the DataFrame.

---

## Section 9: Difficult & Critical Evaluator Questions

### Q36: "Why did you use Logistic Regression instead of modern Deep Learning like BERT or RoBERTa?"
**Answer**: 
1. **Academic Foundations**: CSE276 focuses on foundational AI and machine learning principles (information theory, gradient optimization, loss formulations, and linear decision boundaries), which are best demonstrated through statistical models.
2. **Computational Efficiency**: Our Logistic Regression model achieves **76.43% accuracy** with inference latency under **20 milliseconds** and a memory footprint of less than **500 KB** on standard CPU hardware. BERT requires hundreds of megabytes of VRAM, GPU acceleration, and incurs 100–300 ms latency per query.
3. **Interpretability**: Linear model coefficients directly correspond to specific n-gram weights (e.g., positive weights for `"outstanding"`, negative weights for `"terrible"`), offering transparent explainability.

### Q37: "Are the output probabilities from Softmax true Bayesian calibrated probabilities?"
**Answer**: No. Softmax outputs represent monotonic scores normalized between 0 and 1 via exponentiation. While they indicate model relative confidence, they are not calibrated empirical probabilities unless explicit post-processing (such as Platt Scaling or Isotonic Regression with Brier Score evaluation) is applied. In SENTINEL AI, we accurately refer to them as *Softmax likelihood estimates* rather than calibrated probabilities.

### Q38: "What happens if a user submits text with slang, emojis, or heavy sarcasm?"
**Answer**: 
- **Slang**: If slang words occurred at least 3 times in the training corpus (e.g., `"smh"`, `"ugh"`), they are preserved in the TF-IDF vocabulary and assigned weights.
- **Emojis**: Emojis are currently stripped during regex normalization.
- **Sarcasm**: Sarcasm (e.g., `"Oh great, another 5-hour delay!"`) is a classic challenge for bag-of-words and n-gram models because it uses positive words in negative contexts. While `"5-hour delay"` provides negative bigram signals, sophisticated sarcasm detection requires contextual transformer architectures or acoustic/prosodic cues.

### Q39: "If the SQLite database grows to millions of records, what bottlenecks will emerge?"
**Answer**: SQLite operates with file-level write locking. While read queries scale well with indexes, concurrent write transactions will experience lock contention. For high-volume multi-user production, `DATABASE_URL` would be redirected to a managed PostgreSQL cluster with connection pooling and table partitioning by date.

### Q40: "How did you ensure that test data did not leak into the training process?"
**Answer**: Data split integrity was maintained through strict order-of-operations:
1. Raw dataset was split into 80% train and 20% test using `train_test_split(stratify=y, random_state=42)`.
2. The `TfidfVectorizer` was fitted **exclusively on the training split** (`fit_transform(X_train)`).
3. The test split was transformed using the already-fitted vectorizer (`transform(X_test)`).
4. No statistics, vocabulary, or IDF weights from the test set were accessible during training.

### Q41: "Why did you choose L2 regularization over L1 regularization?"
**Answer**: L1 regularization (Lasso) produces sparse weight vectors by driving non-essential coefficients to zero (feature selection). L2 regularization (Ridge) shrinks all weights proportionally. In NLP sentiment classification with dense n-gram vocabularies, hundreds of minor words each contribute subtle sentiment signals. L2 regularization retains these distributed signals, yielding higher cross-validation accuracy on text classification.

### Q42: "What is Laplace smoothing in Naive Bayes, and what problem does it solve?"
**Answer**: In Naive Bayes, word likelihood is calculated as $P(w \mid c) = \frac{\text{count}(w, c)}{\sum_{w'} \text{count}(w', c)}$. If a word in the test document never appeared in class $c$ in the training set, its probability is 0. Because probabilities are multiplied, a single zero causes the entire document probability to become 0 (the zero-frequency problem). Laplace smoothing adds $\alpha$ (e.g., $\alpha=1.0$) to the numerator and $\alpha \cdot |V|$ to the denominator, ensuring non-zero probabilities for unseen terms.

### Q43: "How does the platform handle oversized batch uploads without running out of memory?"
**Answer**: Uploads are subject to a strict 10 MB file size limit enforced in memory before streaming. For large CSV files, data is processed in vectorized pandas batches, and annotated outputs are written to disk before response delivery.

### Q44: "How do you defend the project against SQL injection?"
**Answer**: The project uses SQLAlchemy ORM which utilizes parameterized queries under the hood. When user search input is passed to `.filter(SentimentAnalysis.input_text.ilike(f"%{search}%"))`, the database engine treats the search string as a literal data parameter rather than executable SQL syntax. This was validated by automated test `test_12_sql_injection_resilience_in_search`.

### Q45: "What was the most challenging bug encountered during full-stack integration, and how was it resolved?"
**Answer**: During Phase 6 integration testing, the test suite failed with `OperationalError: no such table: sentiment_analyses`. Investigation revealed that the unit test module `test_backend.py` applied an isolated test database fixture using `app.dependency_overrides[get_db] = override_get_db`, and dropped its tables on module teardown without clearing the override. Consequently, subsequent test modules inherited the stale override pointing to a dropped database. We resolved this by explicitly adding `app.dependency_overrides.pop(get_db, None)` in fixture cleanup, ensuring 100% test isolation across all 39 tests.
