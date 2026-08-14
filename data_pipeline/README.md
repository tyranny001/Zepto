# Module 1 — Data Pipeline

## Design notes (Sprint 1)

### Data source & scope

- **Source:** [books.toscrape.com](http://books.toscrape.com) — public scraping-practice site, no auth.
- **Scope:** Scrape the first **5 paginated listing pages** of the "All products" catalogue (20 books/page → ≥100 books) and capture each book's **category** from its detail page. This satisfies ≥60 books across ≥3 categories without hard-coding category URLs.

### Schema (normalized SQLite)

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,  -- 0/1 boolean
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

**PK/FK:** `books.category_id → categories.category_id`.

### Cleaning rules

| Field | Raw | Cleaned | Failure handling |
|-------|-----|---------|------------------|
| price | `£51.77` | `price_gbp` float | Median imputation |
| star_rating | `Three` | `rating` int 1–5 | Median imputation |
| availability | `In stock (22 available)` | `in_stock` bool | Drop row (cannot infer) |
| currency | — | `price_inr = price_gbp × 105.50` | Fixed project rate |

**Fixed conversion rate:** 1 GBP = **105.50 INR** (project-defined constant, no API lookup).

### Pipeline layout

```
run_pipeline.py   — orchestrator: scrape → clean → load → query → pandas check
scraper.py        — HTTP fetch + BeautifulSoup parsing
cleaner.py        — type coercion + imputation
db.py             — schema creation + inserts
queries.py        — 5+ SQL queries + pd.read_sql / pd.merge equivalence
books.db          — generated SQLite file (or recreated by run_pipeline.py)
query_output.txt  — logged query results
```

### SQL query plan (≥5, all required clauses)

1. **SELECT/WHERE** — in-stock books with rating ≥ 4
2. **ORDER BY + LIMIT** — top 10 highest-rated books
3. **DISTINCT** — unique categories represented in catalogue
4. **IN** — books in selected category names
5. **JOIN + BETWEEN** — books priced between INR bounds, joined to category names

## Install & run

```bash
cd data_pipeline
pip install -r ../requirements.txt
python run_pipeline.py
```

Outputs: `books.db`, `query_output.txt`, console logs including pandas equivalence check.
