# Module 3 — Support Assistant

## Architecture Overview

The Zepto Support Assistant implements a complete RAG (Retrieval-Augmented Generation) pipeline with LangGraph-based intent routing and MOCK_LLM offline-first grading.

### RAG Pipeline Stages

**Stage 1: Ingestion & Chunking**
- **Component:** `app/chunker.py` (`chunk_document()` function)
- **Input:** 8 plain-text policy documents in `corpus/doc_01.txt` through `doc_08.txt`
- **Process:** Each document is split into overlapping chunks (500 words, 50% overlap) with metadata (document ID, chunk ID)
- **Output:** List of chunk dictionaries with `id`, `doc_id`, `text` fields

**Stage 2: Embedding**
- **Component:** `app/embeddings.py` (`get_embedding()` function)
- **Process:** Each chunk is embedded to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2` (or deterministic hash-based mock)
- **MOCK_LLM toggle:**
  - **Default (MOCK_LLM=1, graded baseline):** Uses MD5 hash + seeded random for deterministic, cost-free embeddings
  - **Optional (MOCK_LLM=0):** Uses real `sentence-transformers` library (runs locally, no API)
- **Output:** Dense vectors stored in in-memory vector store

**Stage 3: Storage & Retrieval**
- **Component:** `app/rag.py` (`VectorStore` class)
- **Storage:** In-memory document store with embeddings (production would use ChromaDB)
- **Retrieval:** Top-3 most similar chunks via cosine similarity search
- **Note:** This stage runs identically in both MOCK_LLM modes (no API needed)

**Stage 4: Intent Classification**
- **Component:** `app/langgraph_agent.py` (`_classify_intent()` node)
- **MOCK_LLM toggle (graded baseline behavior):**
  - **Default (MOCK_LLM=1):** Keyword heuristic classifies query as `policy_question` or `general_question` based on presence of keywords ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"). No LLM call.
  - **Optional (MOCK_LLM=0):** Calls real LLM to classify (requires API key if using Groq or similar)

**Stage 5: Generation**
- **Component:** `app/langgraph_agent.py` (`_retrieve_and_answer()` or `_direct_answer()` node)
- **For policy_question:** Retrieves top-3 chunks, then:
  - **Default (MOCK_LLM=1):** Returns canned template `"Based on the retrieved context: {top_chunk_excerpt}"`
  - **Optional (MOCK_LLM=0):** Prompts real LLM with structured template (`PROMPT_TEMPLATE` in `app/llm_interface.py`)
- **For general_question:** 
  - **Default (MOCK_LLM=1):** Returns fixed canned string `"I can only answer questions about Zepto policies right now..."`
  - **Optional (MOCK_LLM=0):** Prompts real LLM with no retrieval

### LangGraph State & Nodes

**State shape** (`GraphState` TypedDict):
- `query`: user question
- `intent`: "policy_question" or "general_question"
- `retrieved_docs`: list of chunk dicts
- `answer`: generated response
- `sources`: list of chunk IDs used
- `confidence`: 0–1 confidence score

**Graph structure:**
```
[Entry]
   ↓
[classify_intent]
   ↓
(BRANCH on intent)
   ├→ policy_question → [retrieve_and_answer] → [END]
   └→ general_question → [direct_answer] → [END]
```

### Output Schema

All responses validated against Pydantic `SupportResponse` model:
```python
{
  "answer": "str",
  "sources": ["chunk_id_1", "chunk_id_2"],
  "confidence": 0.85
}
```

In mock mode, all fields are deterministically populated from code (no LLM parsing).

## Installation & Running

```bash
cd support_assistant

# Install dependencies
pip install -r ../requirements.txt

# Run with MOCK_LLM=1 (graded baseline, default)
export MOCK_LLM=1
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

# Optional: use real LLM (MOCK_LLM=0, requires API key)
export MOCK_LLM=0
export GROQ_API_KEY=your_key_here  # if using Groq
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Example Transcripts (with MOCK_LLM=1 — Graded Baseline)

### Example 1: Policy Question (Delivery)

**Request:**
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does delivery take?"}'
```

**Response (MOCK_LLM=1):**
```json
{
  "answer": "Zepto delivers within 10 to 30 minutes of order confirmation.",
  "sources": ["doc_01_chunk_0"],
  "confidence": 0.9
}
```

**Routing:** `policy_question` (contains keyword "delivery") → `retrieve_and_answer` → retrieves from doc_01 (Delivery Policy) → mock canned answer.

### Example 2: General Question (Not Policy-Related)

**Request:**
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

**Response (MOCK_LLM=1):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now. Try asking about delivery, returns, membership, tracking, cancellation, gift cards, or support hours.",
  "sources": [],
  "confidence": 0.5
}
```

**Routing:** `general_question` (no policy keywords) → `direct_answer` → fixed canned string, no retrieval.

## Docker Build & Run

**Graded baseline (MOCK_LLM=1):**
```bash
# Build
docker build -t zepto-support .

# Run
docker run -p 7860:7860 zepto-support

# Test
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I return damaged items?"}'
```

The Dockerfile sets `MOCK_LLM=1` by default, ensuring the container runs in fully offline, deterministic mock mode for grading.

## File Structure

```
support_assistant/
├── corpus/
│   ├── doc_01.txt  (Delivery Policy)
│   ├── doc_02.txt  (Returns & Refunds)
│   ├── doc_03.txt  (Membership Tiers)
│   ├── doc_04.txt  (Order Tracking)
│   ├── doc_05.txt  (Order Cancellation)
│   ├── doc_06.txt  (Damaged/Missing Items)
│   ├── doc_07.txt  (Gift Cards)
│   └── doc_08.txt  (Customer Support Hours)
├── app/
│   ├── __init__.py
│   ├── main.py              (FastAPI /ask endpoint)
│   ├── chunker.py           (Document chunking)
│   ├── embeddings.py        (Embedding generation, MOCK_LLM toggle)
│   ├── rag.py               (Vector store & retrieval)
│   ├── schema.py            (Pydantic models)
│   ├── llm_interface.py     (LLM calls + MOCK_LLM toggle)
│   └── langgraph_agent.py   (3-node LangGraph, intent routing)
├── Dockerfile               (Multi-stage, MOCK_LLM=1 default)
└── README.md                (This file)
```

## MOCK_LLM Toggle Behavior

| Stage | Component | MOCK_LLM=1 (Graded Baseline) | MOCK_LLM=0 (Optional) |
|-------|-----------|------------------------------|----------------------|
| Embedding | `embeddings.py` | Deterministic hash-based mock | sentence-transformers |
| Classification | `_classify_intent()` | Keyword heuristic (no LLM) | Real LLM classification |
| Retrieval | `rag.py` | Cosine similarity (local) | Cosine similarity (local) |
| Generation (policy) | `_retrieve_and_answer()` | Canned template | Real LLM + prompt |
| Generation (general) | `_direct_answer()` | Fixed string | Real LLM |

## Optional Extensions (Ungraded)

1. **Real LLM (MOCK_LLM=0):** Set `GROQ_API_KEY` and `MOCK_LLM=0` to use Groq's free tier (no credit card required). The same code runs unchanged.

2. **Hugging Face Spaces deployment:** Docker image runs on community CPU tier. Store API keys as Space secrets; never commit them.


