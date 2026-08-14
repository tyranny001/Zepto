"""Embedding generation using sentence-transformers (or mock fallback)."""

from __future__ import annotations

import hashlib
import os

import numpy as np


def get_embedding(text: str) -> list[float]:
    """Get embedding for text via sentence-transformers or mock.
    
    Default (MOCK_LLM unset or 1): deterministic hash-based mock (384 dims).
    With MOCK_LLM=0: uses real all-MiniLM-L6-v2 model if available.
    """
    use_mock = os.getenv("MOCK_LLM", "1").lower() in ("1", "true")

    if use_mock:
        return mock_embed(text)

    # Try real embedding
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()
    except Exception:
        # Fallback to mock if real unavailable
        return mock_embed(text)


def mock_embed(text: str) -> list[float]:
    """Deterministic mock embedding via MD5 hash + seeded random."""
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.randn(384).tolist()  # 384 dims to match all-MiniLM-L6-v2
