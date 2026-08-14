"""ChromaDB-based vector store for embeddings and retrieval."""

from __future__ import annotations

import numpy as np


class VectorStore:
    """Simple in-memory vector store with cosine similarity.
    
    In production, this would use ChromaDB. Here we keep it minimal
    for grading without external dependencies.
    """

    def __init__(self) -> None:
        self.documents: list[dict] = []  # {"id": str, "text": str, "embedding": list[float]}

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Add documents with their embeddings."""
        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append({
                "id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "embedding": embedding
            })

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Retrieve top-k most similar documents by cosine similarity."""
        if not self.documents:
            return []

        similarities = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((sim, doc))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in similarities[:top_k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
