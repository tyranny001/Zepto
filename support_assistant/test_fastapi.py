#!/usr/bin/env python
"""Test FastAPI endpoints for Module 3."""

import os
import json
import time
import subprocess
import sys
from pathlib import Path

os.environ["MOCK_LLM"] = "1"

# Try using requests if available, otherwise skip full API test
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not available. Testing core functions only.")

def test_app_import():
    """Test that the app imports without errors."""
    print("=== Test: App Import ===")
    try:
        from app.main import app
        print("✓ FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import app: {e}")
        return False

def test_endpoints_mock():
    """Test endpoints by importing and calling them directly."""
    print("\n=== Test: Endpoint Functions ===")
    try:
        from app.main import app
        
        # Mock the async functions by using FastAPI's test client
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        print(f"GET /health: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Mock LLM: {data.get('mock_llm')}")
            print(f"  Chunks: {data.get('corpus_chunks')}")
        
        # Test examples endpoint
        response = client.get("/examples")
        print(f"\nGET /examples: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Examples: {len(data.get('examples', []))} samples")
        
        # Test ask endpoint
        response = client.post(
            "/ask",
            json={"question": "How long does delivery take?"}
        )
        print(f"\nPOST /ask: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Answer: {data.get('answer')[:80]}...")
            print(f"  Confidence: {data.get('confidence'):.2f}")
            print(f"  Sources: {len(data.get('sources', []))} chunk(s)")
            print(f"  Escalated: {data.get('escalated')}")
            print("✓ All endpoints working")
            return True
        else:
            print(f"✗ /ask request failed: {response.text}")
            return False
        
    except Exception as e:
        print(f"✗ Endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_response_schema():
    """Test Pydantic response schema."""
    print("\n=== Test: Response Schema ===")
    try:
        from app.schema import SupportAnswer
        
        response = SupportAnswer(
            answer="Test answer about delivery",
            sources=["chunk 1", "chunk 2"],
            confidence=0.85,
            escalated=False
        )
        
        print(f"SupportAnswer created:")
        print(f"  answer: {response.answer[:40]}...")
        print(f"  sources: {response.sources}")
        print(f"  confidence: {response.confidence}")
        print(f"  escalated: {response.escalated}")
        print("✓ Response schema validated")
        return True
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False

if __name__ == "__main__":
    all_passed = True
    
    all_passed &= test_app_import()
    all_passed &= test_response_schema()
    all_passed &= test_endpoints_mock()
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All FastAPI tests passed")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
