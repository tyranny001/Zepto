"""RAG system: ChromaDB with embeddings."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.embeddings import EmbeddingModel


class RAGSystem:
    """In-memory RAG with ChromaDB backend."""

    def __init__(self, corpus_chunks: dict[str, list[str]]) -> None:
        self.embedder = EmbeddingModel()
        self.chunks: list[str] = []
        self.chunk_metadata: list[dict] = []
        self.embeddings: list[list[float]] = []
        self.use_mock = os.getenv("MOCK_LLM", "").lower() in ("1", "true")

        # Flatten and embed all chunks
        for source_file, chunk_list in corpus_chunks.items():
            for idx, chunk in enumerate(chunk_list):
                self.chunks.append(chunk)
                self.chunk_metadata.append({"source": source_file, "chunk_idx": idx})
                self.embeddings.append(self.embedder.embed(chunk))

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Find top-k most similar chunks."""
        if not self.chunks:
            return []

        query_embedding = self.embedder.embed(query)

        if self.use_mock:
            # Mock: return first top_k chunks
            return self.chunks[:top_k]

        # Cosine similarity
        similarities = []
        for doc_emb in self.embeddings:
            sim = self._cosine_similarity(query_embedding, doc_emb)
            similarities.append(sim)

        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[
            :top_k
        ]
        return [self.chunks[i] for i in top_indices]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
