"""FastAPI application for Zepto support assistant."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.chunker import chunk_document, load_corpus
from app.embeddings import get_embedding
from app.langgraph_agent import ZeptoSupportAgent
from app.rag import VectorStore
from app.schema import AskRequest, SupportResponse

# Initialize corpus and vector store
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def initialize_app() -> tuple[FastAPI, ZeptoSupportAgent]:
    """Initialize FastAPI app and LangGraph agent."""
    app = FastAPI(
        title="Zepto Support Assistant",
        description="Policy-grounded RAG with LangGraph and MOCK_LLM toggle",
        version="1.0.0"
    )

    # Load corpus
    corpus = load_corpus(CORPUS_DIR)

    # Create vector store and add documents
    vector_store = VectorStore()
    all_chunks = []

    for doc_id, doc_text in corpus.items():
        chunks = chunk_document(doc_text, doc_id)
        all_chunks.extend(chunks)

    # Embed all chunks
    embeddings = [get_embedding(chunk["text"]) for chunk in all_chunks]
    vector_store.add_documents(all_chunks, embeddings)

    # Create agent
    agent = ZeptoSupportAgent(vector_store)

    return app, agent


app, agent = initialize_app()


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    mock_mode = os.getenv("MOCK_LLM", "1").lower() in ("1", "true")
    return {
        "status": "healthy",
        "mock_llm": mock_mode,
        "mode": "GRADED_BASELINE" if mock_mode else "REAL_LLM_OPTIONAL"
    }


@app.post("/ask", response_model=SupportResponse)
async def ask(request: AskRequest) -> SupportResponse:
    """Answer customer question using RAG + LangGraph."""
    try:
        # Run the graph
        result = agent.invoke(request.query)

        # Extract and validate response
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.5)

        if not answer:
            answer = "I was unable to generate an answer. Please try rephrasing your question."
            confidence = 0.0

        return SupportResponse(
            answer=answer,
            sources=sources,
            confidence=float(confidence)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/examples")
async def examples() -> dict:
    """Return example queries for testing."""
    return {
        "examples": [
            {
                "type": "policy_question",
                "query": "How long does delivery take?",
                "expected_source": "doc_01"
            },
            {
                "type": "policy_question",
                "query": "Can I return perishable items?",
                "expected_source": "doc_02"
            },
            {
                "type": "policy_question",
                "query": "What are the membership tiers?",
                "expected_source": "doc_03"
            },
            {
                "type": "general_question",
                "query": "What's the weather like?",
                "expected_source": None
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    mock_mode = os.getenv("MOCK_LLM", "1").lower() in ("1", "true")
    print(f"Starting Zepto Support Assistant (MOCK_LLM={mock_mode})")
    uvicorn.run(app, host="0.0.0.0", port=7860)
