# Zepto AI/ML Capstone

End-to-end analytics and ML capstone for Zepto's analytics guild: scrape and normalize catalog data, run Titanic analytics/ML pipelines, and deploy an offline-capable support assistant.

## Repository structure

```
/data_pipeline   — Module 1: scrape → clean → SQLite (25 marks)
/analytics       — Module 2: Titanic EDA + ML (50 marks)
/support_assistant — Module 3: RAG + LangGraph + FastAPI (25 marks)
requirements.txt — Consolidated Python dependencies (all modules)
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Requirements strategy

**One consolidated `requirements.txt` at the repo root** pins dependencies for all three modules. Rationale:

- Single `pip install` for graders and CI
- Avoids version drift between modules
- Module READMEs document any module-specific env vars (e.g. `MOCK_LLM=1`)

## Fixed project constants

| Constant | Value | Used in |
|----------|-------|---------|
| GBP → INR conversion | **1 GBP = 105.50 INR** | Module 1 (`price_inr`) |
| Titanic cache | `analytics/titanic.csv` | Module 2 (load once from `sns.load_dataset`) |
| LLM default | `MOCK_LLM=1` (offline) | Module 3 |

## Module run instructions

### Module 1 — Data Pipeline (`/data_pipeline`)

<!-- Filled in during Sprint 2 -->

```bash
cd data_pipeline
python run_pipeline.py
```

### Module 2 — Analytics (`/analytics`)

<!-- Filled in during Sprints 3–4 -->

```bash
cd analytics
jupyter notebook titanic_analysis.ipynb
```

### Module 3 — Support Assistant (`/support_assistant`)

<!-- Filled in during Sprint 5 -->

```bash
cd support_assistant
set MOCK_LLM=1
uvicorn app.main:app --reload
```

## Design decisions (summary)

| Module | Key decisions |
|--------|---------------|
| **Data Pipeline (Module 1)** | Scrape books.toscrape.com 5 listing pages (100+ books, 25+ categories). Two-table PK/FK schema (categories ↔ books). Fixed conversion: 1 GBP = 105.50 INR. ≥5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN, JOIN. Pandas equivalence check validates join results match SQL. |
| **Analytics (Module 2)** | Load Titanic once via sns.load_dataset(), save as titanic.csv offline fallback. Part A: profile, clean (missing-value thresholds), EDA with ≥4 interpreted charts, correlation matrix (6 specified columns), standardization check. Part B: stratified split, leak-safe ColumnTransformer pipeline, 3 classifiers (DT/RF/LR), full metric suite, 3-way imbalance comparison (baseline/class_weight/SMOTE), GridSearchCV + OOB, regression side-task (fare prediction), joblib pipeline save/reload. |
| **Support Assistant (Module 3)** | 8 Zepto policy documents, chunked per-document, embedded with all-MiniLM-L6-v2 (384 dims). LangGraph 3-node agent: classify_intent (keyword heuristic MOCK_LLM=1, optional LLM MOCK_LLM=0), retrieve_and_answer (top-3 via cosine sim), direct_answer (canned strings). Structured prompt template with 5 skeleton components + negative constraint + few-shot. Pydantic schema (answer/sources/confidence). FastAPI /ask endpoint. MOCK_LLM=1 (graded baseline, fully offline) vs MOCK_LLM=0 (optional real LLM). |

## Git workflow

Feature branch `feature/capstone-build` carries incremental work; merged to `main` at release (Sprint 7).

## License

Educational capstone submission.
