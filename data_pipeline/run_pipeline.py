"""End-to-end data pipeline: scrape → clean → SQLite → query → pandas check."""

from __future__ import annotations

from pathlib import Path

from cleaner import clean_books
from db import init_db, load_dataframe, read_tables
from queries import compare_join_results, run_queries
from scraper import scrape_books

DB_PATH = Path(__file__).resolve().parent / "books.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "query_output.txt"


def main() -> None:
    print("Step 1: Scraping books.toscrape.com (5 catalogue pages)...")
    raw = scrape_books(max_pages=5)
    print(f"  Scraped {len(raw)} raw book records.")

    print("Step 2: Cleaning and converting currency (1 GBP = 105.50 INR)...")
    df = clean_books(raw)
    n_categories = df["category"].nunique()
    print(f"  Clean rows: {len(df)} | Categories: {n_categories}")
    print(f"  Columns: {list(df.columns)} | dtypes:\n{df.dtypes}")

    if len(df) < 60:
        raise SystemExit(f"FAIL: need >= 60 books, got {len(df)}")
    if n_categories < 3:
        raise SystemExit(f"FAIL: need >= 3 categories, got {n_categories}")

    print("Step 3: Loading into SQLite...")
    conn = init_db(DB_PATH)
    load_dataframe(conn, df)
    cats, books = read_tables(conn)
    print(f"  categories: {len(cats)} rows | books: {len(books)} rows")

    print("Step 4: Running SQL queries...")
    run_queries(conn, OUTPUT_PATH)
    print(f"  Query output saved to {OUTPUT_PATH.name}")

    print("Step 5: pd.read_sql vs pd.merge equivalence check...")
    sql_df, merge_df, equivalent = compare_join_results(conn)
    print("\n--- SQL join result (head) ---")
    print(sql_df.head(10).to_string(index=False))
    print("\n--- pandas merge result (head) ---")
    print(merge_df.head(10).to_string(index=False))
    print(f"\nEquivalence: {'PASS' if equivalent else 'FAIL'}")
    conn.close()

    if not equivalent:
        raise SystemExit("Join equivalence check failed.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
