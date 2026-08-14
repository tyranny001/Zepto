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

<!-- Per-module summaries added during Sprint 7 -->

| Module | Key decisions |
|--------|---------------|
| Data Pipeline | _TBD_ |
| Analytics | _TBD_ |
| Support Assistant | _TBD_ |

## Git workflow

Feature branch `feature/capstone-build` carries incremental work; merged to `main` at release (Sprint 7).

## License

Educational capstone submission.
