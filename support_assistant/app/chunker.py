"""Chunk corpus files into overlapping text segments."""

from __future__ import annotations

import re
from pathlib import Path


def chunk_file(file_path: Path, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split file into overlapping chunks by word count."""
    text = file_path.read_text(encoding="utf-8")

    # Split by words
    words = re.findall(r"\b\w+\b|[.,!?;:\-]", text)

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_corpus(corpus_dir: Path) -> dict[str, list[str]]:
    """Load and chunk all .md files in corpus directory."""
    corpus = {}
    markdown_files = sorted(corpus_dir.glob("*.md"))
    for md_file in markdown_files:
        file_key = md_file.stem
        corpus[file_key] = chunk_file(md_file)
    return corpus
