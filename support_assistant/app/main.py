"""FastAPI app for Zepto support assistant."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chunker import load_corpus
from app.graph import create_graph
from app.rag import RAGSystem
from app.schema import SupportAnswer

# Initialize corpus and RAG
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
corpus = load_corpus(CORPUS_DIR)
rag = RAGSystem(corpus)
graph = create_graph(rag)

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-powered customer support with offline MOCK_LLM mode",
    version="1.0.0",
)


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""

    question: str = Field(..., min_length=3, max_length=500, description="Customer question")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    mock_mode = os.getenv("MOCK_LLM", "").lower() in ("1", "true")
    return {
        "status": "healthy",
        "mock_llm": mock_mode,
        "corpus_chunks": sum(len(chunks) for chunks in corpus.values()),
    }


@app.post("/ask", response_model=SupportAnswer)
async def ask(request: AskRequest) -> SupportAnswer:
    """Answer customer question using RAG + LangGraph."""
    try:
        # Run graph
        initial_state = {
            "question": request.question,
            "retrieved_chunks": [],
            "draft_answer": "",
            "final_answer": "",
            "route": "",
            "confidence": 0.5,
            "attempt_count": 0,
        }

        result = graph.invoke(initial_state)

        # Extract results
        final_answer = result.get("final_answer", "")
        confidence = result.get("confidence", 0.5)
        retrieved_chunks = result.get("retrieved_chunks", [])
        route = result.get("route", "answer")

        # Format sources (chunk previews)
        sources = [chunk[:50] + "..." for chunk in retrieved_chunks]

        # Determine if escalated
        escalated = route == "escalate"

        return SupportAnswer(
            answer=final_answer or "I'm unable to help with this query. Please contact our support team.",
            sources=sources,
            confidence=float(confidence),
            escalated=escalated,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/examples")
async def examples() -> dict:
    """Return example Q&A pairs demonstrating the assistant."""
    return {
        "examples": [
            {
                "question": "How long does Zepto delivery take?",
                "expected_topic": "delivery timeframe",
            },
            {
                "question": "What's your return policy for fresh items?",
                "expected_topic": "returns and fresh groceries",
            },
            {
                "question": "Can I use multiple payment methods for one order?",
                "expected_topic": "payments",
            },
            {
                "question": "How do I track my order?",
                "expected_topic": "delivery tracking",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn

    mock_mode = os.getenv("MOCK_LLM", "").lower() in ("1", "true")
    print(f"Starting Zepto Support Assistant (MOCK_LLM={'ON' if mock_mode else 'OFF'})")
    uvicorn.run(app, host="0.0.0.0", port=8000)
