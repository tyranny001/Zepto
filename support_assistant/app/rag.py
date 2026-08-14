"""ChromaDB-based vector store for document embeddings and retrieval."""

from __future__ import annotations

from pathlib import Path

import chromadb


# Persist directory for ChromaDB (inside support_assistant/)
CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")


def get_chroma_collection(collection_name: str = "zepto_policies") -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = chromadb.Client(chromadb.Settings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False,
    ))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


class VectorStore:
    """ChromaDB-backed vector store for Zepto policy documents.

    Embeds and stores document chunks, retrieves top-k by cosine similarity.
    ChromaDB runs entirely locally — no API key, no network needed.
    """

    def __init__(self, collection_name: str = "zepto_policies") -> None:
        self.collection = get_chroma_collection(collection_name)

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Add document chunks with their embeddings to ChromaDB."""
        if not chunks:
            return

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"doc_id": chunk["doc_id"]} for chunk in chunks]

        # Upsert to avoid duplicates on re-run
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Retrieve top-k most similar documents by cosine similarity from ChromaDB."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs
