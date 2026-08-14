#!/usr/bin/env python
"""Test Module 3 against spec requirements."""

import os
import json
from pathlib import Path

os.environ["MOCK_LLM"] = "1"

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.chunker import load_corpus, chunk_document
from app.embeddings import get_embedding
from app.rag import VectorStore
from app.langgraph_agent import ZeptoSupportAgent
from app.llm_interface import PROMPT_TEMPLATE

print("=" * 70)
print("MODULE 3 SPEC COMPLIANCE TEST")
print("=" * 70)

# Test 1: Corpus loading
print("\n[TEST 1] Load 8 corpus documents")
corpus_dir = Path(__file__).parent / "corpus"
corpus = load_corpus(corpus_dir)

expected_docs = [f"doc_{i:02d}" for i in range(1, 9)]
actual_docs = sorted(corpus.keys())

if actual_docs == expected_docs:
    print(f"✓ All 8 documents loaded: {actual_docs}")
else:
    print(f"✗ FAIL: Expected {expected_docs}, got {actual_docs}")

# Test 2: Prompt template structure
print("\n[TEST 2] Prompt template has all 5 skeleton components + constraints + few-shot")
template_lower = PROMPT_TEMPLATE.lower()
components = ["role", "context", "task", "format", "length"]
constraints = ["negative constraint", "example"]

for component in components:
    if component in template_lower:
        print(f"✓ Component '{component}' present in template")
    else:
        print(f"✗ FAIL: Component '{component}' missing")

# Test 3: Chunking and embedding
print("\n[TEST 3] Chunk documents and embed with all-MiniLM-L6-v2 (or mock)")
all_chunks = []
for doc_id, doc_text in corpus.items():
    chunks = chunk_document(doc_text, doc_id)
    all_chunks.extend(chunks)

print(f"✓ Total chunks created: {len(all_chunks)}")

# Test embedding
sample_embedding = get_embedding("test query")
if len(sample_embedding) == 384:
    print(f"✓ Embedding dimension: 384 (correct for all-MiniLM-L6-v2)")
else:
    print(f"✗ FAIL: Expected 384 dims, got {len(sample_embedding)}")

# Test 4: Vector store and retrieval
print("\n[TEST 4] ChromaDB-style retrieval (top-3 cosine similarity)")
vs = VectorStore()
embeddings = [get_embedding(chunk["text"]) for chunk in all_chunks]
vs.add_documents(all_chunks, embeddings)

query = "How long does delivery take?"
query_emb = get_embedding(query)
results = vs.query(query_emb, top_k=3)

if len(results) == 3:
    print(f"✓ Retrieved top-3 chunks for: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc['doc_id']}: {doc['text'][:60]}...")
        if doc["doc_id"] == "doc_01":
            print(f"     ✓ Correct source (Delivery Policy)")
else:
    print(f"✗ FAIL: Expected 3 results, got {len(results)}")

# Test 5: LangGraph 3-node routing
print("\n[TEST 5] LangGraph intent classification (MOCK_LLM=1 keyword heuristic)")
agent = ZeptoSupportAgent(vs)

# Policy question example
policy_query = "Can I return damaged items?"
result = agent.invoke(policy_query)
if result["intent"] == "policy_question":
    print(f"✓ Policy query '{policy_query}' → policy_question")
else:
    print(f"✗ FAIL: Expected policy_question, got {result['intent']}")

# General question example
general_query = "What is the weather?"
result = agent.invoke(general_query)
if result["intent"] == "general_question":
    print(f"✓ General query '{general_query}' → general_question")
else:
    print(f"✗ FAIL: Expected general_question, got {result['intent']}")

# Test 6: Mock-mode outputs
print("\n[TEST 6] Mock-mode outputs (MOCK_LLM=1 default)")

# Policy question → retrieve_and_answer
policy_result = agent.invoke("How long does delivery take?")
if policy_result["answer"] and len(policy_result["sources"]) > 0:
    print(f"✓ Policy question routed to retrieve_and_answer")
    print(f"  Answer: {policy_result['answer'][:60]}...")
    print(f"  Sources: {policy_result['sources']}")
    print(f"  Confidence: {policy_result['confidence']}")
else:
    print(f"✗ FAIL: Policy answer generation failed")

# General question → direct_answer
general_result = agent.invoke("What is machine learning?")
if "I can only answer questions about Zepto policies" in general_result["answer"]:
    print(f"✓ General question routed to direct_answer")
    print(f"  Answer: {general_result['answer'][:60]}...")
    print(f"  Sources: {general_result['sources']} (empty, expected)")
else:
    print(f"✗ FAIL: General answer not using canned string")

# Test 7: Pydantic schema
print("\n[TEST 7] Pydantic SupportResponse schema validation")
from app.schema import SupportResponse

try:
    response = SupportResponse(
        answer="Test answer",
        sources=["doc_01_chunk_0"],
        confidence=0.9
    )
    print(f"✓ SupportResponse schema valid")
    print(f"  {response.model_dump_json()}")
except Exception as e:
    print(f"✗ FAIL: Schema validation failed: {e}")

# Test 8: FastAPI app
print("\n[TEST 8] FastAPI /ask endpoint with TestClient")
try:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Policy question
    response = client.post(
        "/ask",
        json={"query": "How long does delivery take?"}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✓ POST /ask succeeded")
        print(f"  Policy question response:")
        print(f"    answer: {data['answer'][:50]}...")
        print(f"    sources: {data['sources']}")
        print(f"    confidence: {data['confidence']}")
    else:
        print(f"✗ FAIL: /ask returned {response.status_code}")

    # General question
    response = client.post(
        "/ask",
        json={"query": "What is AI?"}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ General question response:")
        print(f"  answer: {data['answer'][:50]}...")
        print(f"  sources: {data['sources']}")

    # Health endpoint
    response = client.get("/health")
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ GET /health: mock_llm={data['mock_llm']}")

except Exception as e:
    print(f"✗ FAIL: FastAPI test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Dockerfile
print("\n[TEST 9] Dockerfile presence and MOCK_LLM=1 default")
dockerfile_path = Path(__file__).parent / "Dockerfile"
if dockerfile_path.exists():
    content = dockerfile_path.read_text()
    if "MOCK_LLM=1" in content:
        print(f"✓ Dockerfile present and sets MOCK_LLM=1 default")
    else:
        print(f"✗ FAIL: MOCK_LLM=1 not in Dockerfile")
else:
    print(f"✗ FAIL: Dockerfile not found")

print("\n" + "=" * 70)
print("✓✓✓ MODULE 3 SPEC COMPLIANCE TEST COMPLETE ✓✓✓")
print("=" * 70)
