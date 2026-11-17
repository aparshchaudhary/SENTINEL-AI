# SENTINEL AI – Academic Demonstration & Presentation Guide

**Academic Course**: CSE276 – Artificial Intelligence Foundations  
**Target Evaluation**: Final Viva & Project Demonstration  
**Platform**: SENTINEL AI – Intelligent Sentiment Analysis Platform  

---

## 1. Demonstration Prerequisites

Before commencing the demonstration, ensure the host machine satisfies:
- **Python**: 3.10+ installed and `.venv` active.
- **Node.js**: 18+ installed with npm.
- **Terminal Shells**: 2 open terminal windows (one for Backend, one for Frontend).
- **Web Browser**: Google Chrome, Mozilla Firefox, or Microsoft Edge.

---

## 2. Server Startup & Verification

### Terminal 1: Launch FastAPI Backend
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI"
.\.venv\Scripts\Activate.ps1
python backend\main.py
```
**Expected Console Output**:
```
======================================================================
STARTING SENTINEL AI BACKEND SERVICE...
Project: SENTINEL AI – Intelligent Sentiment Analysis Platform (v1.0.0)
======================================================================
[DATABASE] SQLite schema tables initialized successfully.
[ML ENGINE] Model pre-loaded: Multinomial Logistic Regression (L2, Balanced)
[SENTINEL AI] Backend is ready to accept requests.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Terminal 2: Launch React Frontend
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI\frontend"
npm run dev
```
**Expected Console Output**:
```
  VITE v5.4.21  ready in 450 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## 3. Recommended 6-Step Demonstration Sequence

### Step 1: Health Diagnostics & Swagger Documentation
1. Open browser to **`http://127.0.0.1:8000/docs`**.
2. Show the evaluators the auto-generated **Swagger UI**.
3. Expand `GET /api/v1/health` &rarr; click **Try it out** &rarr; **Execute**.
4. Point out the live response confirming:
   - `status: "healthy"`
   - `database: "connected"`
   - `model_loaded: true`
   - `active_model: "Multinomial Logistic Regression (L2, Balanced)"`

---

### Step 2: Live Sentiment Analyzer (Interactive Inference)
1. Navigate browser to **`http://localhost:5173/analyzer`**.
2. Point out the text input area with character countdown (max 5,000 characters).
3. Test **Positive Sentiment**:
   - Click the prompt button **"Positive Experience"** or paste:  
     `"I absolutely love this airline! The flight crew was outstanding, seats were very comfortable, and we arrived 20 minutes early."`
   - Click **Analyze Sentiment** (or press `Ctrl+Enter`).
   - Highlight the **Positive Badge**, **~89% Confidence**, latency (**<20 ms**), and animated green probability bar.
   - Click **"NLP Preprocessed Tokens"** to reveal tokenized features:  
     `"absolutely love airline flight crew outstanding seat comfortable arrived 20 minute early"`
4. Test **Negative Sentiment with Negation Preservation**:
   - Paste: `"The flight was not good and the staff was never helpful."`
   - Click **Analyze Sentiment**.
   - Show that the system correctly predicts **Negative** with high confidence.
   - Open **"NLP Preprocessed Tokens"** to highlight bound negation tokens:  
     `"flight not_good staff never_helpful"`  
     *Explain to evaluator: Traditional NLP strips 'not' and 'never', causing false positive classifications. SENTINEL AI preserves negation context.*
5. Test **Neutral Statement**:
   - Click **"Neutral Statement"** or paste:  
     `"The scheduled flight UA104 departs at 4:15 PM from terminal 3, gate B12."`
   - Click **Analyze Sentiment**.
   - Show predicted **Neutral** polarity.

---

### Step 3: High-Throughput Batch CSV Processor
1. Navigate to **`http://localhost:5173/batch`**.
2. Create or use a small CSV dataset (e.g. `test_feedback.csv`):
   ```csv
   id,feedback_text
   1,Wonderful flight experience with friendly staff!
   2,Terrible delay and rude gate agent. Never flying again.
   3,Flight scheduled for 10 AM departed on time.
   ```
3. Drag and drop the CSV into the upload dropzone.
4. Show the file size and name detection.
5. Click **Start Batch Inference**.
6. Point out:
   - Fast processing duration (**<0.2 seconds**).
   - Real-time distribution breakdown cards (Positive, Negative, Neutral percentages).
   - Instant preview table showing annotated rows.
7. Click **Download Annotated CSV**.
8. Open the downloaded file in Excel/Notepad to demonstrate attached columns:
   `sentiment_predicted`, `confidence_score`, `prob_positive`, `prob_negative`, `prob_neutral`.

---

### Step 4: Executive Dashboard & Analytics
1. Navigate to **`http://localhost:5173/`**.
2. Show that telemetry numbers updated dynamically:
   - **Total Analyses**: Real-time count of records evaluated.
   - **Positive & Negative Counts**: Proportions dynamically reflecting recent predictions.
   - **Average Model Confidence**: Softmax likelihood average.
3. Hover over the **Sentiment Activity Timeline** (Area Chart) to show daily volume.
4. Hover over the **Polarity Distribution** (Donut Chart) to show categorical breakdown.
5. Point out that all charts fetch live data from `/api/v1/analytics/` with zero hardcoding.

---

### Step 5: Audit History & Persistence
1. Navigate to **`http://localhost:5173/history`**.
2. Show the paginated table displaying recent queries with exact timestamps and latency.
3. Demonstrate **Search**: Type `"customer service"` &rarr; table instantly filters to matching records.
4. Demonstrate **Polarity Filter**: Select `"Negative Only"` from dropdown &rarr; only negative remarks display.
5. Demonstrate **Record Deletion**: Click the trash icon next to a test record &rarr; confirm deletion in the modal &rarr; record is deleted from SQLite.

---

### Step 6: Model Performance & Confusion Matrix
1. Navigate to **`http://localhost:5173/model`**.
2. Walk the evaluator through the machine learning benchmarks:
   - **Holdout Test Accuracy**: **76.43%** on 2,881 unseen samples.
   - **Macro F1-Score**: **0.7138** (unweighted harmonic mean).
   - **TF-IDF Vocabulary Size**: **7,235 features** (unigrams + bigrams).
   - **Hyperparameters**: $C=0.5$, balanced class weighting, `lbfgs` solver.
3. Review the **Confusion Matrix**:
   - Diagonal numbers represent correct classifications.
   - Explain that balanced class weighting prevents the model from collapsing onto the majority negative class.

---

## 4. Common Errors & Troubleshooting During Demo

| Issue | Quick Fix |
| :--- | :--- |
| **"API Offline" badge in header** | Verify Terminal 1 is running `python backend\main.py`. Test `http://127.0.0.1:8000/api/v1/health` in browser. |
| **Port 8000 already in use** | Run `netstat -ano | findstr :8000` in PowerShell and terminate the stale process (`taskkill /F /PID <pid>`). |
| **CSV Upload Error: "Could not detect text column"** | Ensure the CSV has a column named `text`, `review`, `comment`, or `feedback_text`, or type the column name in the optional input field. |
| **Vite Dev Server Port Changed to 5174** | Ensure port 5173 was free before running `npm run dev`. Both 5173 and 5174 are whitelisted in CORS. |
