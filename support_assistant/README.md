# Module 3 — Support Assistant

## Design notes (Sprint 1)

### Architecture

```
User → FastAPI /ask → LangGraph (retrieve → generate → validate) → Pydantic response
                              ↑
                         ChromaDB (local embeddings)
```

### LangGraph state shape

```python
class GraphState(TypedDict):
    question: str
    retrieved_chunks: list[str]
    draft_answer: str
    final_answer: str
    route: str          # "answer" | "escalate"
    confidence: float
```

### Graph nodes (3 + conditional routing)

1. **retrieve** — embed question, query ChromaDB top-k chunks
2. **generate** — LLM (or MOCK_LLM) with structured prompt
3. **validate** — Pydantic schema check; route to END or back to generate

**Conditional edge:** if `confidence < 0.5` or answer fails validation → regenerate (max 1 retry) else → END.

### Pydantic response schema

```python
class SupportAnswer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    escalated: bool
```

### Prompt template structure

- **Role:** Zepto grocery support agent
- **Context:** retrieved FAQ chunks
- **Task:** answer customer question
- **Format:** JSON matching `SupportAnswer`
- **Length:** ≤150 words
- **Negative constraint:** do not invent refund policies not in context
- **Few-shot:** 2 example Q&A pairs from corpus

### Offline default (`MOCK_LLM=1`)

- No network calls for LLM or embeddings at grade time
- Mock returns deterministic answers from retrieved chunks
- Real OpenAI/Ollama path optional via env vars

### Corpus

- `corpus/faq_delivery.md`, `corpus/faq_returns.md`, `corpus/faq_payments.md`
- Chunk size 300 tokens, overlap 50
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (cached locally after first run) or hash-based mock vectors when offline

### Deployment

- `Dockerfile` with `MOCK_LLM=1` default
- `uvicorn app.main:app --host 0.0.0.0 --port 8000`
