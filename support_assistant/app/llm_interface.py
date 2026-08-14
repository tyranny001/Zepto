"""LLM interface with MOCK_LLM toggle and structured prompt template."""

from __future__ import annotations

import json
import os
from typing import Optional


# ===== STRUCTURED PROMPT TEMPLATE (Used when MOCK_LLM=0) =====
PROMPT_TEMPLATE = """You are a Zepto customer support assistant. Your role is to provide accurate, helpful answers to customer questions based only on the provided policy documents.

CONTEXT:
{context}

TASK:
Answer the following customer question using only the information in the context above. If the answer is not found in the context, say so clearly.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the provided context. Do not make up policies, fees, or procedures that are not explicitly stated.

FORMAT:
Respond in valid JSON with exactly these fields:
- answer (string): Your response
- confidence (float, 0.0 to 1.0): How confident you are

LENGTH:
Keep your answer concise, under 150 words.

FEW-SHOT EXAMPLES:

Example 1:
Q: How long does delivery take?
A: {{"answer": "Zepto delivers within 10 to 30 minutes of order confirmation, depending on your delivery zone and current order volume.", "confidence": 0.95}}

Example 2:
Q: Can I return spoiled milk?
A: {{"answer": "Yes, grocery and perishable items can be returned within 24 hours of delivery if damaged or spoiled. Report it through the 'Report an Issue' button.", "confidence": 0.90}}

CUSTOMER QUESTION:
{question}

RESPONSE:
"""


class LLMInterface:
    """LLM interface with MOCK_LLM toggle for grading baseline."""

    def __init__(self) -> None:
        self.use_mock = os.getenv("MOCK_LLM", "1").lower() in ("1", "true")

    def answer_question(
        self,
        question: str,
        context: str,
        max_retries: int = 2
    ) -> dict:
        """Generate answer given question and context.
        
        Returns dict with 'answer', 'confidence' keys.
        """
        if self.use_mock:
            return self._mock_answer(question, context)

        # Optional: call real LLM (e.g., Groq)
        return self._real_llm_answer(question, context, max_retries)

    @staticmethod
    def _mock_answer(question: str, context: str) -> dict:
        """Mock answer based on keyword heuristics."""
        q_lower = question.lower()

        # Simple keyword matching
        if "delivery" in q_lower:
            return {
                "answer": "Zepto delivers within 10 to 30 minutes of order confirmation.",
                "confidence": 0.9
            }
        elif "return" in q_lower or "refund" in q_lower:
            return {
                "answer": "Perishable items can be returned within 24 hours if damaged. Non-perishables can be returned within 7 days if unopened.",
                "confidence": 0.85
            }
        elif "membership" in q_lower:
            return {
                "answer": "Zepto offers three tiers: Basic (free), Pass (₹49/month), and Pass+ (₹99/month) with increasing benefits.",
                "confidence": 0.85
            }
        elif "tracking" in q_lower or "track" in q_lower:
            return {
                "answer": "Your order shows a live rider-tracking map from packing until delivery on the 'Track Order' screen.",
                "confidence": 0.9
            }
        elif "cancel" in q_lower:
            return {
                "answer": "Orders can be cancelled free of cost before the status changes to 'Packed', typically within 2 minutes.",
                "confidence": 0.9
            }
        elif "gift card" in q_lower:
            return {
                "answer": "Gift cards are available in denominations of ₹100, ₹250, ₹500, and ₹1000, valid for 1 year.",
                "confidence": 0.85
            }
        elif "support" in q_lower or "help" in q_lower:
            return {
                "answer": "Zepto support is available 24/7 via in-app chat with average response time under 2 minutes.",
                "confidence": 0.9
            }

        # Default fallback
        return {
            "answer": "Based on the provided context, I can help answer questions about Zepto's delivery, returns, membership, tracking, cancellation, gift cards, and support policies.",
            "confidence": 0.6
        }

    def _real_llm_answer(
        self,
        question: str,
        context: str,
        max_retries: int = 2
    ) -> dict:
        """Call real LLM (e.g., Groq) with structured prompt."""
        try:
            import requests
        except ImportError:
            # Fall back to mock if requests unavailable
            return self._mock_answer(question, context)

        prompt = PROMPT_TEMPLATE.format(question=question, context=context)

        # Try Groq API (example; adjust to your preferred provider)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return self._mock_answer(question, context)

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "mixtral-8x7b-32768",
                        "messages": [
                            {"role": "system", "content": "You are a JSON-generating assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 256,
                    },
                    timeout=10,
                )

                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    result = json.loads(content)
                    if "answer" in result and "confidence" in result:
                        return result

            except (json.JSONDecodeError, KeyError, requests.RequestException):
                if attempt < max_retries:
                    continue
                break

        # All retries failed; fall back to mock
        return self._mock_answer(question, context)
