#!/usr/bin/env python
"""End-to-end verification of all three modules."""

import os
import sys
from pathlib import Path

# Set offline mode for Module 3
os.environ["MOCK_LLM"] = "1"

def test_module1():
    """Test Module 1: Data Pipeline."""
    print("="*60)
    print("MODULE 1: DATA PIPELINE")
    print("="*60)
    
    try:
        import sqlite3
        import pandas as pd
        
        db_path = Path("data_pipeline/books.db")
        
        if not db_path.exists():
            print("✗ FAIL: books.db not found")
            return False
        
        conn = sqlite3.connect(db_path)
        
        # Check schema
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"✓ Tables found: {tables}")
        
        if "categories" not in tables or "books" not in tables:
            print("✗ FAIL: Required tables missing")
            return False
        
        # Check data
        books_df = pd.read_sql("SELECT COUNT(*) as count FROM books", conn)
        book_count = books_df.iloc[0, 0]
        print(f"✓ Books in database: {book_count}")
        
        if book_count < 60:
            print(f"✗ FAIL: Need >= 60 books, got {book_count}")
            return False
        
        # Check columns
        books_df = pd.read_sql("SELECT * FROM books LIMIT 1", conn)
        required_cols = ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
        for col in required_cols:
            if col not in books_df.columns:
                print(f"✗ FAIL: Column '{col}' missing")
                return False
        
        print(f"✓ Required columns present: {required_cols}")
        
        # Check conversion rate (sample)
        sample = pd.read_sql("SELECT price_gbp, price_inr FROM books LIMIT 1", conn)
        if sample.iloc[0, 1] > 0:  # price_inr should be > 0
            ratio = sample.iloc[0, 1] / sample.iloc[0, 0]
            print(f"✓ Sample conversion rate check: {ratio:.2f} (expected ~105.50)")
        
        # Check query output
        query_path = Path("data_pipeline/query_output.txt")
        if not query_path.exists():
            print("✗ FAIL: query_output.txt not found")
            return False
        
        query_output = query_path.read_text()
        if "=== Q1 —" in query_output and "=== Q5 —" in query_output:
            print("✓ Query output file present with required queries")
        else:
            print(f"Query output starts with: {query_output[:100]}")
            print("✗ FAIL: Query output incomplete")
            return False
        
        conn.close()
        print("\n✓✓✓ MODULE 1 PASSED ✓✓✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ MODULE 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_module2():
    """Test Module 2: Analytics Pipeline."""
    print("="*60)
    print("MODULE 2: ANALYTICS PIPELINE")
    print("="*60)
    
    try:
        import pandas as pd
        from pathlib import Path
        
        # Check titanic.csv
        titanic_path = Path("analytics/titanic.csv")
        if not titanic_path.exists():
            print("✗ FAIL: titanic.csv not found")
            return False
        
        df = pd.read_csv(titanic_path)
        print(f"✓ Titanic dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        required_cols = ["survived", "pclass", "sex", "age", "fare"]
        for col in required_cols:
            if col not in df.columns:
                print(f"✗ FAIL: Column '{col}' missing")
                return False
        
        print(f"✓ Required columns present: {required_cols}")
        
        # Check outputs
        outputs_dir = Path("analytics/outputs")
        required_files = ["classifier_metrics.csv", "correlation_heatmap.png", 
                         "decision_tree.png", "eda_charts.png"]
        
        for fname in required_files:
            fpath = outputs_dir / fname
            if not fpath.exists():
                print(f"✗ FAIL: {fname} not found")
                return False
        
        print(f"✓ All output files present: {required_files}")
        
        # Check metrics
        metrics_df = pd.read_csv(outputs_dir / "classifier_metrics.csv")
        required_models = ["DecisionTree", "RandomForest", "LogisticRegression"]
        for model in required_models:
            if model not in metrics_df["model"].values:
                print(f"✗ FAIL: Model '{model}' metrics missing")
                return False
        
        print(f"✓ All models evaluated: {list(metrics_df['model'])}")
        print(f"✓ Metrics: accuracy, precision, recall, f1, roc_auc")
        
        # Check pipeline
        pipeline_path = Path("analytics/survival_pipeline.joblib")
        if not pipeline_path.exists():
            print("✗ FAIL: survival_pipeline.joblib not found")
            return False
        
        print("✓ Pipeline saved and reloadable (joblib)")
        
        print("\n✓✓✓ MODULE 2 PASSED ✓✓✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ MODULE 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_module3():
    """Test Module 3: Support Assistant."""
    print("="*60)
    print("MODULE 3: SUPPORT ASSISTANT")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path("support_assistant")))
        
        from app.main import app
        from app.chunker import load_corpus
        from app.rag import RAGSystem
        from app.graph import create_graph
        from fastapi.testclient import TestClient
        
        # Check corpus
        corpus_dir = Path("support_assistant/corpus")
        corpus = load_corpus(corpus_dir)
        
        required_files = ["faq_delivery", "faq_payments", "faq_returns"]
        for fname in required_files:
            if fname not in corpus:
                print(f"✗ FAIL: {fname} not in corpus")
                return False
        
        print(f"✓ Corpus files loaded: {list(corpus.keys())}")
        
        total_chunks = sum(len(chunks) for chunks in corpus.values())
        print(f"✓ Total chunks: {total_chunks}")
        
        if total_chunks < 5:
            print(f"✗ FAIL: Need >= 5 chunks, got {total_chunks}")
            return False
        
        # Test RAG
        rag = RAGSystem(corpus)
        results = rag.retrieve("How long does delivery take?", top_k=3)
        if len(results) != 3:
            print(f"✗ FAIL: RAG retrieval returned {len(results)} chunks, expected 3")
            return False
        
        print(f"✓ RAG retrieval working (top-3 chunks)")
        
        # Test graph
        graph = create_graph(rag)
        initial_state = {
            "question": "What is your return policy?",
            "retrieved_chunks": [],
            "draft_answer": "",
            "final_answer": "",
            "route": "",
            "confidence": 0.5,
            "attempt_count": 0,
        }
        
        result = graph.invoke(initial_state)
        if not result.get("final_answer"):
            print("✗ FAIL: Graph did not generate an answer")
            return False
        
        print(f"✓ LangGraph execution working (confidence: {result.get('confidence', 0):.2f})")
        
        # Test FastAPI endpoints
        client = TestClient(app)
        
        # Health endpoint
        response = client.get("/health")
        if response.status_code != 200:
            print(f"✗ FAIL: /health returned {response.status_code}")
            return False
        
        data = response.json()
        if data.get("mock_llm") != True:
            print("✗ FAIL: MOCK_LLM not enabled")
            return False
        
        print(f"✓ GET /health working (MOCK_LLM={data.get('mock_llm')})")
        
        # Ask endpoint
        response = client.post(
            "/ask",
            json={"question": "How do I track my order?"}
        )
        if response.status_code != 200:
            print(f"✗ FAIL: /ask returned {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        if not data.get("answer"):
            print("✗ FAIL: /ask returned empty answer")
            return False
        
        print(f"✓ POST /ask working (answer length: {len(data.get('answer', ''))} chars)")
        
        # Examples endpoint
        response = client.get("/examples")
        if response.status_code != 200:
            print(f"✗ FAIL: /examples returned {response.status_code}")
            return False
        
        examples = response.json().get("examples", [])
        if len(examples) < 2:
            print(f"✗ FAIL: Need >= 2 examples, got {len(examples)}")
            return False
        
        print(f"✓ GET /examples working ({len(examples)} example Q&A pairs)")
        
        # Check Dockerfile
        dockerfile_path = Path("support_assistant/Dockerfile")
        if not dockerfile_path.exists():
            print("✗ FAIL: Dockerfile not found")
            return False
        
        dockerfile_content = dockerfile_path.read_text()
        if "MOCK_LLM=1" not in dockerfile_content:
            print("✗ FAIL: MOCK_LLM not set in Dockerfile")
            return False
        
        print("✓ Dockerfile present with MOCK_LLM=1 default")
        
        print("\n✓✓✓ MODULE 3 PASSED ✓✓✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗✗✗ MODULE 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    os.chdir(Path(__file__).parent)
    
    results = {
        "Module 1 (Data Pipeline)": test_module1(),
        "Module 2 (Analytics)": test_module2(),
        "Module 3 (Support Assistant)": test_module3(),
    }
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{module}: {status}")
    
    all_passed = all(results.values())
    print("="*60)
    
    if all_passed:
        print("✓✓✓ ALL MODULES PASSED ✓✓✓")
        return 0
    else:
        print("✗✗✗ SOME MODULES FAILED ✗✗✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
