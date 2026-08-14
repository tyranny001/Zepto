"""SQL queries and pandas equivalence checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

QUERIES: list[tuple[str, str]] = [
    (
        "Q1 — SELECT/WHERE: in-stock books rated 4 or 5",
        """
        SELECT title, rating, in_stock
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, title
        LIMIT 15
        """,
    ),
    (
        "Q2 — ORDER BY + LIMIT: top 10 highest-rated books",
        """
        SELECT title, rating, price_inr
        FROM books
        ORDER BY rating DESC, price_inr ASC
        LIMIT 10
        """,
    ),
    (
        "Q3 — DISTINCT: unique categories in catalogue",
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
        """,
    ),
    (
        "Q4 — IN: books in Poetry, Travel, or Mystery categories",
        """
        SELECT b.title, c.category_name, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Poetry', 'Travel', 'Mystery')
        ORDER BY c.category_name, b.title
        LIMIT 20
        """,
    ),
    (
        "Q5 — JOIN + BETWEEN: mid-range INR prices with category names",
        """
        SELECT b.title, c.category_name, b.price_inr, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.price_inr BETWEEN 500 AND 3000
          AND b.in_stock = 1
        ORDER BY b.price_inr DESC
        LIMIT 15
        """,
    ),
]

JOIN_QUERY = """
SELECT b.title, c.category_name, b.price_inr, b.rating, b.in_stock
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE b.rating >= 3
ORDER BY b.rating DESC, b.price_inr ASC
"""


def run_queries(conn: sqlite3.Connection, output_path: Path) -> None:
    lines: list[str] = []
    for label, sql in QUERIES:
        df = pd.read_sql(sql, conn)
        lines.append(f"=== {label} ===")
        lines.append(sql.strip())
        lines.append(df.to_string(index=False))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def compare_join_results(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    sql_df = pd.read_sql(JOIN_QUERY, conn)
    categories = pd.read_sql("SELECT * FROM categories", conn)
    books = pd.read_sql("SELECT * FROM books", conn)

    merge_df = books.merge(categories, on="category_id", how="inner")
    merge_df = merge_df[merge_df["rating"] >= 3][
        ["title", "category_name", "price_inr", "rating", "in_stock"]
    ]
    merge_df = merge_df.sort_values(
        ["rating", "price_inr"], ascending=[False, True]
    ).reset_index(drop=True)

    sql_sorted = sql_df.sort_values(
        ["rating", "price_inr"], ascending=[False, True]
    ).reset_index(drop=True)

    equivalent = sql_sorted.equals(merge_df)
    return sql_sorted, merge_df, equivalent
