#!/usr/bin/env python
"""Test Module 3 components."""

import os
import json
from pathlib import Path

os.environ["MOCK_LLM"] = "1"

from app.chunker import load_corpus
from app.rag import RAGSystem
from app.graph import create_graph
from app.schema import SupportAnswer

def test_corpus_loading():
    print("=== Test 1: Corpus Loading ===")
    corpus_dir = Path("corpus")
    corpus = load_corpus(corpus_dir)
    print(f"Corpus files: {list(corpus.keys())}")
    for name, chunks in corpus.items():
        print(f"  {name}: {len(chunks)} chunks")
    assert len(corpus) == 3, "Expected 3 corpus files"
    print("✓ Corpus loaded successfully\n")
    return corpus

def test_rag_retrieval(corpus):
    print("=== Test 2: RAG Retrieval ===")
    rag = RAGSystem(corpus)
    
    queries = [
        "How long does delivery take?",
        "Can I return fresh items?",
        "What payment methods do you accept?",
    ]
    
    for query in queries:
        results = rag.retrieve(query, top_k=3)
        print(f"Query: \"{query}\"")
        print(f"  Retrieved {len(results)} chunks")
        for i, chunk in enumerate(results, 1):
            preview = chunk[:60].replace("\n", " ") + "..."
            print(f"    Chunk {i}: {preview}")
    
    print("✓ RAG retrieval working\n")
    return rag

def test_graph_execution(rag):
    print("=== Test 3: LangGraph Execution ===")
    graph = create_graph(rag)
    
    test_questions = [
        "How long does Zepto delivery take?",
        "What's your return policy for fresh items?",
        "Can I use multiple payment methods?",
    ]
    
    for question in test_questions:
        initial_state = {
            "question": question,
            "retrieved_chunks": [],
            "draft_answer": "",
            "final_answer": "",
            "route": "",
            "confidence": 0.5,
            "attempt_count": 0,
        }
        
        result = graph.invoke(initial_state)
        
        print(f"Q: {question}")
        print(f"A: {result.get('final_answer', 'No answer')[:100]}...")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Escalated: {result.get('route') == 'escalate'}")
        print()
    
    print("✓ LangGraph execution working\n")

if __name__ == "__main__":
    corpus = test_corpus_loading()
    rag = test_rag_retrieval(corpus)
    test_graph_execution(rag)
    print("=== All Module 3 tests passed ===")
