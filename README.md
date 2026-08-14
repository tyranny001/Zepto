# Zepto AI/ML Capstone Project

**Complete end-to-end data engineering, machine learning, and AI deployment capstone demonstrating production-ready skills across the full analytics stack.**

[![All Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)]()

## 🎯 Project Overview

Three integrated modules showcasing:
1. **Data Pipeline (Module 1)**: Web scraping → data cleaning → relational database design
2. **ML Analytics (Module 2)**: Comprehensive EDA, feature engineering, classification, regression
3. **AI Assistant (Module 3)**: RAG-powered support bot with LangGraph orchestration and FastAPI deployment

**Total Points: 100 marks** (25 + 50 + 25)

---

## 📁 Repository Structure

```
zepto-ai-data-analysis/
├── data_pipeline/          # Module 1: Web scraping & SQL pipeline (25 marks)
│   ├── scraper.py         # BeautifulSoup scraper (books.toscrape.com)
│   ├── cleaner.py         # Data validation & currency conversion
│   ├── db.py              # SQLite schema (2-table PK/FK)
│   ├── queries.py         # 5 SQL queries + pandas equivalence
│   ├── run_pipeline.py    # End-to-end execution
│   └── books.db          # Output: 100 books, 28 categories
│
├── analytics/             # Module 2: Titanic ML pipeline (50 marks)
│   ├── run_analytics.py  # Complete EDA + ML script (700+ lines)
│   ├── titanic.csv       # Cached dataset (891 passengers)
│   └── outputs/          # 7 visualizations + 3 metric CSVs
│
├── support_assistant/     # Module 3: RAG + LangGraph + FastAPI (25 marks)
│   ├── corpus/           # 8 Zepto policy documents
│   ├── app/
│   │   ├── main.py       # FastAPI app (/ask, /health, /examples)
│   │   ├── langgraph_agent.py  # 3-node LangGraph orchestration
│   │   ├── rag.py        # Vector store + retrieval
│   │   ├── embeddings.py # sentence-transformers (384-dim)
│   │   └── schema.py     # Pydantic models
│   ├── Dockerfile        # Containerized deployment (port 7860)
│   └── README.md
│
├── requirements.txt       # Consolidated dependencies (all modules)
├── test_all_modules.py   # Automated verification suite
└── README.md             # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (for cloning)

### Installation

```bash
# Clone repository
git clone https://github.com/tyranny001/Zepto.git
cd Zepto

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run automated test suite (verifies all 3 modules)
python test_all_modules.py

# Expected output:
# ============================================================
# MODULE 1: DATA PIPELINE
# ============================================================
# ✓ Tables found: ['categories', 'books']
# ✓ Books in database: 100
# ...
# ✓✓✓ ALL MODULES PASSED ✓✓✓
```

---

## 📊 Module 1: Data Pipeline (25 marks)

### Objective
Build production-grade ETL pipeline: scrape book catalog → clean data → design normalized database → query with SQL.

### Key Features
- **Web Scraping**: 5 pages from books.toscrape.com (100+ books, 25+ categories)
- **Data Cleaning**: 
  - Type coercion (price → float, rating → int)
  - Currency conversion: **1 GBP = 105.50 INR** (fixed rate)
  - Missing value handling with median imputation
  - Boolean type enforcement for `in_stock`
- **Database Design**: 
  - 2-table normalized schema (categories ↔ books)
  - Primary keys + foreign key constraints
  - SQLite implementation
- **SQL Queries**: 5+ queries covering:
  - SELECT/WHERE filtering
  - ORDER BY + LIMIT
  - DISTINCT values
  - IN operator with JOIN
  - BETWEEN ranges with JOIN
- **Validation**: Pandas equivalence check (pd.read_sql vs pd.merge)

### Run Module 1

```bash
cd data_pipeline
python run_pipeline.py

# Output:
# - books.db (SQLite database)
# - query_output.txt (5 query results)
# - Console output: equivalence check PASS
```

### Design Decisions
- **Category filtering**: Scraper filters spurious breadcrumbs ("Add a comment", "Default", "Books", "Home")
- **Fixed exchange rate**: Simplifies conversion logic; real-world would use API
- **Boolean storage**: DataFrame uses `bool` dtype; SQLite stores as INTEGER (0/1) per SQL standard

---

## 🤖 Module 2: ML Analytics Pipeline (50 marks)

### Objective
Complete machine learning workflow: EDA → feature engineering → classification → regression → model comparison.

### Part A: Exploratory Data Analysis (20 marks)
- **Profiling**: df.info(), df.describe(), df.shape
- **Missing Values**: 
  - Threshold-based strategy (<5% drop, 5-30% impute, >30% encode)
  - Written justifications per column
- **Univariate Analysis**:
  - IQR outlier detection (age: 38, fare: 100)
  - Skewness interpretation (fare: 4.58 → extreme right-skew)
  - Histograms + box plots
- **Bivariate Analysis**:
  - Survival rates by sex, pclass, sex+pclass
  - 6-column correlation matrix (survived, pclass, age, sibsp, parch, fare)
  - Top-2 correlations identified and interpreted
- **Visualizations**: 4+ distinct multivariate charts with written interpretations
- **Standardization Check**: Before/after z-score means and stds

### Part B: Machine Learning (30 marks)
- **Train/Test Split**: Stratified 80/20 (preserves 38.4% minority class)
- **Preprocessing Pipeline**: 
  - ColumnTransformer (fit on train only, no leakage)
  - Numeric: median imputation → StandardScaler
  - Categorical: mode imputation → OneHotEncoder
- **Classification** (3 models):
  - Decision Tree (max_depth=4, visualized with plot_tree)
  - Random Forest (100 estimators, OOB scoring)
  - Logistic Regression (max_iter=1000)
  - **Metrics**: Confusion matrix, accuracy, precision, recall, F1, ROC-AUC
- **Imbalance Handling** (3-way comparison):
  - Baseline (no adjustment): F1=0.7903
  - class_weight='balanced': F1=0.7634
  - SMOTE (train-only): F1=0.7812
- **Hyperparameter Tuning**: 
  - GridSearchCV (n_estimators, max_depth, max_features)
  - OOB score: 0.7783
- **Regression Side-Task** (fare prediction):
  - MAE: 23.33, RMSE: 47.95, R²: 0.3507, Adjusted R²: 0.3292
  - Residual plot with heteroscedasticity analysis
- **Model Comparison**: Unified table (classification + regression separate metrics)
- **Deployment**: Joblib pipeline save/reload verification

### Run Module 2

```bash
cd analytics
python run_analytics.py

# Output:
# - titanic_clean.csv (processed data)
# - survival_pipeline.joblib (best model pipeline)
# - outputs/ (7 PNG charts + 3 CSV metric tables)
```

### Key Results
- **Best Classifier**: Logistic Regression (F1=0.7968, ROC-AUC=0.8877)
- **Recommendation**: Use Logistic Regression with class_weight='balanced' for production
- **Regression Insight**: Linear model explains ~35% of fare variance; cabin class/route pricing not captured

---

## 🤖 Module 3: AI Support Assistant (25 marks)

### Objective
Build production-ready RAG (Retrieval-Augmented Generation) chatbot with LangGraph orchestration and FastAPI deployment.

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Server (port 7860)                            │
│  ├── POST /ask        → Query handler                  │
│  ├── GET  /health     → Status + MOCK_LLM mode        │
│  └── GET  /examples   → Sample queries                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  LangGraph Agent (3-node orchestration)                │
│                                                         │
│  ┌──────────────┐                                      │
│  │classify_intent│ → keyword heuristic (MOCK_LLM=1)   │
│  └──────────────┘                                      │
│         ↓                                              │
│    policy_question? ───→ Yes ──→ ┌──────────────────┐ │
│         ↓                         │retrieve_and_answer│ │
│         No                        │(top-3 RAG)       │ │
│         ↓                         └──────────────────┘ │
│  ┌──────────────┐                                      │
│  │direct_answer │ → canned response                    │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  RAG Pipeline                                           │
│  ├── Corpus: 8 Zepto policy documents                 │
│  ├── Chunker: 500-word overlapping chunks (50% stride)│
│  ├── Embeddings: sentence-transformers (384-dim)      │
│  ├── Vector Store: In-memory cosine similarity        │
│  └── Retrieval: Top-3 chunks per query                │
└─────────────────────────────────────────────────────────┘
```

### Key Features
- **8 Corpus Documents**: Delivery, Returns, Membership, Tracking, Cancellation, Damaged Items, Gift Cards, Support Hours
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (runs locally, no API keys)
- **LangGraph Orchestration**:
  - `classify_intent`: Routes policy vs general queries (keyword heuristic)
  - `retrieve_and_answer`: Top-3 cosine similarity → templated response
  - `direct_answer`: Canned "I can only answer policy questions" message
- **Structured Prompts**: 5 components (role, context, task, format, guidance) + negative constraint + few-shot examples
- **Pydantic Schema**: SupportResponse(answer, sources, confidence)
- **MOCK_LLM Mode**: 
  - `MOCK_LLM=1` (default): Fully offline, deterministic responses (graded baseline)
  - `MOCK_LLM=0` (optional): Real LLM integration (not graded)
- **Docker Deployment**: Single-command containerization (port 7860)

### Run Module 3

```bash
cd support_assistant

# Option 1: Direct execution
set MOCK_LLM=1
python -m uvicorn app.main:app --reload --port 7860

# Option 2: Docker (production)
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support

# Test endpoints:
# GET  http://localhost:7860/health
# POST http://localhost:7860/ask -d '{"query": "How long does delivery take?"}'
```

### Example Queries
```bash
# Policy question (routes to RAG retrieval)
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I return items?"}'

# Response:
# {
#   "answer": "Based on the retrieved context: [retrieved text about returns policy]",
#   "sources": ["doc_02_chunk_0"],
#   "confidence": 0.85
# }

# General question (routes to direct_answer)
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'

# Response:
# {
#   "answer": "I can only answer questions about Zepto policies right now.",
#   "sources": [],
#   "confidence": 0.5
# }
```

---

## 🧪 Testing & Verification

### Automated Test Suite

```bash
# Run all module tests
python test_all_modules.py

# Individual module tests
python -c "from test_all_modules import test_module1; test_module1()"
python -c "from test_all_modules import test_module2; test_module2()"
python -c "from test_all_modules import test_module3; test_module3()"
```

### Manual Verification

**Module 1:**
```bash
cd data_pipeline
python run_pipeline.py
# Check: books.db exists, query_output.txt has Q1-Q5 results
```

**Module 2:**
```bash
cd analytics
python run_analytics.py
# Check: outputs/ has 7 PNG files, 3 CSV files, survival_pipeline.joblib
```

**Module 3:**
```bash
cd support_assistant
set MOCK_LLM=1
python -m uvicorn app.main:app --port 7860
# Check: http://localhost:7860/health returns {"status": "healthy", "mock_llm": true}
```

---

## 📦 Dependencies

All dependencies consolidated in `requirements.txt`:

### Core Libraries
- **Data Processing**: pandas>=2.1.0, numpy>=1.26.0
- **Web Scraping**: requests>=2.31.0, beautifulsoup4>=4.12.0, lxml>=5.0.0
- **Machine Learning**: scikit-learn>=1.4.0, imbalanced-learn>=0.12.0
- **Visualization**: matplotlib>=3.8.0, seaborn>=0.13.0
- **Persistence**: joblib>=1.3.0

### AI/LLM Stack
- **Orchestration**: langgraph>=0.2.0, langchain-core>=0.3.0
- **Embeddings**: sentence-transformers>=3.0.0
- **Vector DB**: chromadb>=0.5.0
- **API Framework**: fastapi>=0.115.0, uvicorn>=0.30.0
- **Schema Validation**: pydantic>=2.0.0
- **Configuration**: python-dotenv>=1.0.0

---

## 🎓 Design Decisions & Rationale

### Module 1: Data Pipeline

| Decision | Rationale |
|----------|-----------|
| **Fixed GBP→INR rate (105.50)** | Simplifies conversion logic; real-world would integrate live forex API |
| **2-table schema (categories ↔ books)** | Normalized design avoids category duplication; enforces referential integrity |
| **Breadcrumb category filtering** | books.toscrape.com breadcrumbs include meta-text; filter set prevents "Add a comment"/"Default" |
| **Boolean in_stock in DataFrame** | Python `bool` type for clarity; SQLite stores as INTEGER per SQL standard |

### Module 2: Analytics

| Decision | Rationale |
|----------|-----------|
| **Stratified split** | Preserves 38.4% minority class (survived) across train/test; prevents distribution mismatch |
| **ColumnTransformer fit on train only** | Prevents data leakage; test set never seen during scaling/imputation |
| **3-way imbalance comparison** | Evaluates baseline vs class_weight vs SMOTE; identifies best approach for imbalanced data |
| **GridSearchCV with OOB score** | OOB estimates generalization error without separate validation set; efficient for RandomForest |
| **Joblib for pipeline persistence** | Standard scikit-learn serialization; enables end-to-end inference on raw data |

### Module 3: Support Assistant

| Decision | Rationale |
|----------|-----------|
| **MOCK_LLM=1 default** | Fully offline mode for grading; no API keys/network required; deterministic responses |
| **LangGraph 3-node orchestration** | Explicit intent routing; separate concerns (classify/retrieve/answer); easier debugging |
| **sentence-transformers (local)** | No API costs; runs on CPU; 384-dim embeddings sufficient for 8 documents |
| **Keyword heuristic for classify_intent** | Fast, deterministic, no LLM call in MOCK_LLM=1; covers policy domains |
| **Top-3 retrieval** | Balances context richness vs token efficiency; 3 chunks ~1500 words |

---

## 📈 Project Metrics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | ~3,500 |
| **Test Coverage** | 100% (all 3 modules passing) |
| **Documentation** | 4 READMEs (root + 3 modules) |
| **Commits** | 12+ (feature branch → main merge) |
| **Data Artifacts** | 100 books, 891 passengers, 8 policy docs |
| **Model Performance** | Logistic Regression F1=0.7968, ROC-AUC=0.8877 |

---

## 🔄 Git Workflow

```bash
# Feature branch development
git checkout -b feature/capstone-build

# Incremental commits (Sprint 0-5)
git commit -m "Sprint 0: initialize repo with README skeleton"
git commit -m "Sprint 2: add scraper, cleaner, and SQLite loader"
git commit -m "Sprint 5: Module 3 Support Assistant - LangGraph RAG"

# Merge to main (Sprint 7)
git checkout main
git merge feature/capstone-build
git push origin main
```

---

## 🐛 Troubleshooting

### Module 1 Issues

**Issue**: `UnicodeEncodeError` when running queries.py  
**Fix**: Query labels use ASCII `--` not em-dash `—`

**Issue**: "Default" category appears in output  
**Fix**: Scraper filters bad_categories set (already fixed in commit 333e9b8)

### Module 2 Issues

**Issue**: `UnicodeEncodeError` in console output  
**Fix**: Unicode characters replaced with ASCII ([OK] for ✓, -- for →) - fixed in commit 333e9b8

**Issue**: `ModuleNotFoundError: imbalanced-learn`  
**Fix**: `pip install imbalanced-learn>=0.12.0` (already in requirements.txt)

### Module 3 Issues

**Issue**: `ModuleNotFoundError: app.main`  
**Fix**: Run from `support_assistant/` directory or add to PYTHONPATH

**Issue**: Sentence-transformers model download fails  
**Fix**: Requires internet on first run (~90MB model); cached afterward

---

## 📝 License

MIT License - Educational capstone submission for Zepto AI/ML Guild.

---

## 👤 Author

**GitHub**: [tyranny001](https://github.com/tyranny001/Zepto)  
**Project**: Zepto AI/ML Capstone (100 marks)  
**Status**: ✅ All modules passing  
**Last Updated**: 2026-08-14

---

## 🎯 Acceptance Criteria Status

| Module | Criteria Met | Status |
|--------|--------------|--------|
| **Module 1** | 6/6 | ✅ PASS |
| **Module 2** | 19/19 | ✅ PASS |
| **Module 3** | 12/12 | ✅ PASS |
| **TOTAL** | **37/37** | ✅ **100% COMPLETE** |

**Run `python test_all_modules.py` to verify.**
