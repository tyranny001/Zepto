"""Load and chunk corpus documents."""

from pathlib import Path


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Load all .txt documents from corpus directory.
    
    Returns: dict mapping doc_id (e.g., "doc_01") to full document text.
    """
    corpus = {}
    for txt_file in sorted(corpus_dir.glob("doc_*.txt")):
        doc_id = txt_file.stem  # e.g., "doc_01"
        text = txt_file.read_text(encoding="utf-8")
        corpus[doc_id] = text
    return corpus


def chunk_document(doc_text: str, doc_id: str, chunk_size: int = 500) -> list[dict]:
    """Simple per-document chunking: yield overlapping chunks.
    
    Each chunk is a dict with 'id', 'doc_id', and 'text' keys.
    """
    words = doc_text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size // 2):
        chunk_words = words[i : i + chunk_size]
        if chunk_words:
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            chunks.append({
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text
            })
    
    return chunks

