"""Fetch and parse book listings from books.toscrape.com."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
HEADERS = {"User-Agent": "ZeptoCapstoneBot/1.0 (educational scraping)"}
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


@dataclass
class RawBook:
    title: str
    price: str
    star_rating: str
    availability: str
    category: str


def _get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def _parse_listing_page(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Return detail-page URLs from a catalogue listing page."""
    links: list[str] = []
    for article in soup.select("article.product_pod"):
        anchor = article.select_one("h3 a")
        if anchor and anchor.get("href"):
            links.append(urljoin(page_url, anchor["href"]))
    return links


def _parse_book_detail(url: str, session: requests.Session) -> RawBook:
    soup = _get_soup(url, session)
    title = soup.select_one(".product_main h1")
    price = soup.select_one(".price_color")
    rating_tag = soup.select_one("p.star-rating")
    availability = soup.select_one(".availability")
    breadcrumb = soup.select_one(".breadcrumb")

    category = "Unknown"
    if breadcrumb:
        crumbs = [a.get_text(strip=True) for a in breadcrumb.select("a")]
        if len(crumbs) >= 2:
            category = crumbs[-1]

    star_class = ""
    if rating_tag and rating_tag.get("class"):
        for cls in rating_tag["class"]:
            if cls in RATING_MAP:
                star_class = cls
                break

    return RawBook(
        title=title.get_text(strip=True) if title else "",
        price=price.get_text(strip=True) if price else "",
        star_rating=star_class,
        availability=availability.get_text(strip=True) if availability else "",
        category=category,
    )


def scrape_books(max_pages: int = 5, pause_seconds: float = 0.3) -> list[RawBook]:
    """Scrape first `max_pages` of the All-products catalogue."""
    session = requests.Session()
    books: list[RawBook] = []
    for page_num in range(1, max_pages + 1):
        page_url = BASE_URL if page_num == 1 else urljoin(
            BASE_URL, f"catalogue/page-{page_num}.html"
        )
        soup = _get_soup(page_url, session)
        detail_urls = _parse_listing_page(soup, page_url)
        for detail_url in detail_urls:
            books.append(_parse_book_detail(detail_url, session))
            time.sleep(pause_seconds)

        next_link = soup.select_one("li.next a")
        if not next_link and page_num < max_pages:
            break

    return books
