"""Clean raw scraped books into typed records."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from scraper import RATING_MAP, RawBook

GBP_TO_INR = 105.50


@dataclass
class CleanBook:
    title: str
    price_gbp: float
    price_inr: float
    rating: int
    in_stock: bool
    category: str


def _parse_price_gbp(raw: str) -> float | None:
    match = re.search(r"([\d.]+)", raw.replace(",", ""))
    if not match:
        return None
    return float(match.group(1))


def _parse_in_stock(raw: str) -> bool | None:
    text = raw.lower().strip()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def clean_books(raw_books: list[RawBook]) -> pd.DataFrame:
    """Return cleaned DataFrame; impute numeric failures, drop unparseable availability."""
    rows: list[dict] = []
    for book in raw_books:
        price = _parse_price_gbp(book.price)
        rating = RATING_MAP.get(book.star_rating)
        in_stock = _parse_in_stock(book.availability)

        if in_stock is None:
            continue

        rows.append(
            {
                "title": book.title.strip(),
                "price_gbp_raw": price,
                "rating_raw": rating,
                "in_stock": in_stock,
                "category": book.category.strip() or "Unknown",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No books survived availability parsing.")

    price_median = df["price_gbp_raw"].median()
    rating_median = df["rating_raw"].median()

    df["price_gbp"] = df["price_gbp_raw"].fillna(price_median)
    df["rating"] = df["rating_raw"].fillna(rating_median).astype(int)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]
