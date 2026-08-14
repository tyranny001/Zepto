# Module 3 — Support Assistant

## Design notes (Sprint 1)

### Architecture

```
User → FastAPI /ask → LangGraph (retrieve → generate → validate) → Pydantic response
                              ↑
                         RAGSystem + Embeddings (local, mock-safe)
```

### LangGraph state shape

```python
class GraphState(TypedDict):
    question: str
    retrieved_chunks: list[str]
    draft_answer: str
    final_answer: str
    route: str          # "answer" | "escalate" | "regenerate"
    confidence: float
    attempt_count: int
```

### Graph nodes (3 + conditional routing)

1. **retrieve** — embed question, query RAG top-k chunks (3 by default)
2. **generate** — LLM (or MOCK_LLM) with structured JSON prompt
3. **validate** — confidence threshold check; route to END, regenerate, or escalate

**Conditional edge:** if `confidence < 0.6` and `attempt_count < 2` → regenerate, else route to END (escalate if low confidence).

### Pydantic response schema

```python
class SupportAnswer(BaseModel):
    answer: str             # ≤150 words
    sources: list[str]      # chunk previews
    confidence: float       # 0.0–1.0
    escalated: bool         # True if routed to human
```

### Prompt template structure

- **Role:** Zepto grocery customer support agent
- **Context:** retrieved FAQ chunks (top 3)
- **Task:** answer customer question accurately
- **Format:** Valid JSON with keys: answer, confidence, escalated
- **Length constraint:** answer ≤150 words
- **Negative constraint:** do NOT invent refund policies, discounts, or terms not in FAQ
- **Fallback:** if uncertain, set escalated=true

### Offline default (`MOCK_LLM=1`)

- No network calls for LLM or embeddings when offline mode is enabled
- Mock LLM: deterministic keyword-based responses
- Mock embeddings: MD5 hash + seeded numpy randomness (reproducible, corpus-agnostic)
- Real OpenAI/Ollama path optional via `OPENAI_API_KEY` env var

### Corpus and chunking

- **Files:** `corpus/faq_delivery.md`, `corpus/faq_returns.md`, `corpus/faq_payments.md`
- **Chunk strategy:** 300 words per chunk, 50-word overlap
- **Retrieval:** top-k=3 chunks via cosine similarity (or mock top-3 fallback)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384 dims) or mock MD5-seeded (384 dims)

### Deployment

- **FastAPI endpoints:**
  - `GET /health` — status + mock_llm flag + corpus size
  - `POST /ask` — main query endpoint
  - `GET /examples` — sample Q&A pairs
- **Dockerfile:** multi-stage, Python 3.11-slim, `MOCK_LLM=1` default
- **Run local:** `uvicorn app.main:app --reload` (auto-enables debug mode)
- **Run prod:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Install & run

```bash
cd support_assistant

# Offline mode (default, no network)
export MOCK_LLM=1
uvicorn app.main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long does delivery take?"}'

# Docker build & run
docker build -t zepto-support .
docker run -p 8000:8000 zepto-support
```

## Example transcripts (Sprint 5 output)

### Example 1: Delivery timeframe

**Request:**
```json
{
  "question": "How long does Zepto delivery take?"
}
```

**Response (MOCK_LLM=1):**
```json
{
  "answer": "Zepto offers 10-30 minute delivery. Delivery is free on orders above ₹199-₹299. Our delivery hours are 6 AM to 11 PM.",
  "sources": [
    "Delivery timeframe question and answer...",
    "Delivery hours information..."
  ],
  "confidence": 0.8,
  "escalated": false
}
```

### Example 2: Fresh item returns

**Request:**
```json
{
  "question": "Can I return milk or bread if it's damaged?"
}
```

**Response (MOCK_LLM=1):**
```json
{
  "answer": "Yes, fresh items like milk and bread are returnable if damaged or defective at delivery. Report the issue within 24 hours. We'll redeliver or provide a full refund.",
  "sources": [
    "Fresh item return policy excerpt...",
    "Return initiation process..."
  ],
  "confidence": 0.75,
  "escalated": false
}
```
