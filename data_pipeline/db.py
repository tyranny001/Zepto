"""SQLite schema creation and data loading."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    return conn


def load_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    categories = sorted(df["category"].unique())
    category_rows = [(idx + 1, name) for idx, name in enumerate(categories)]
    conn.executemany(
        "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
        category_rows,
    )
    cat_map = {name: cid for cid, name in category_rows}

    book_rows = [
        (
            idx + 1,
            row.title,
            float(row.price_gbp),
            float(row.price_inr),
            int(row.rating),
            int(row.in_stock),
            cat_map[row.category],
        )
        for idx, row in enumerate(df.itertuples(index=False))
    ]
    conn.executemany(
        """
        INSERT INTO books (book_id, title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        book_rows,
    )
    conn.commit()


def read_tables(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories = pd.read_sql("SELECT * FROM categories", conn)
    books = pd.read_sql("SELECT * FROM books", conn)
    return categories, books
