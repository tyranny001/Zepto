"""Embeddings: sentence-transformers (or mock for MOCK_LLM mode)."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np


class EmbeddingModel:
    """Embedding interface with fallback to mock."""

    def __init__(self, use_mock: bool = False) -> None:
        self.use_mock = use_mock or os.getenv("MOCK_LLM", "").lower() in ("1", "true")
        self.model = None
        if not self.use_mock:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Failed to load sentence-transformers: {e}. Falling back to mock.")
                self.use_mock = True

    def embed(self, text: str) -> list[float]:
        """Return embedding for text."""
        if self.use_mock:
            return self._mock_embed(text)
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for multiple texts."""
        if self.use_mock:
            return [self._mock_embed(t) for t in texts]
        return self.model.encode(texts).tolist()

    @staticmethod
    def _mock_embed(text: str) -> list[float]:
        """Deterministic mock embedding via hash + seeded random."""
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(384).tolist()  # 384 dims to match MiniLM
