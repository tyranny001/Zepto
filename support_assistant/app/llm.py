"""LLM interface with offline MOCK_LLM fallback."""

from __future__ import annotations

import json
import os
from typing import Any


class LLMInterface:
    """LLM with mock and real (optional) backends."""

    def __init__(self) -> None:
        self.use_mock = os.getenv("MOCK_LLM", "").lower() in ("1", "true")
        if not self.use_mock:
            try:
                import openai

                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            except Exception:
                print("Warning: OpenAI not available. Using MOCK_LLM.")
                self.use_mock = True

    def call(self, prompt: str, system: str = "", max_tokens: int = 500) -> str:
        """Call LLM (or mock) with prompt."""
        if self.use_mock:
            return self._mock_call(prompt, system)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}. Falling back to mock.")
            return self._mock_call(prompt, system)

    @staticmethod
    def _mock_call(prompt: str, system: str = "") -> str:
        """Mock LLM response based on prompt content."""
        # Simple rule-based mock
        lower_prompt = prompt.lower()

        mock_responses = {
            "delivery": "Zepto offers 10-30 minute delivery. Delivery is free on orders above ₹199-₹299. Our delivery hours are 6 AM to 11 PM.",
            "return": "Returns are accepted within 7 days for unopened items. Fresh items are non-returnable unless damaged. Refunds process in 5-7 business days.",
            "refund": "Refunds are processed within 5-7 business days to your original payment method. Report issues within 24 hours of delivery.",
            "payment": "We accept credit/debit cards, UPI, net banking, and digital wallets. All payments are encrypted and secure.",
            "discount": "Use coupon codes at checkout if eligible. Order value must meet the minimum requirement. Check the Offers section for current deals.",
        }

        for keyword, response in mock_responses.items():
            if keyword in lower_prompt:
                confidence = 0.8
                return json.dumps(
                    {
                        "answer": response,
                        "confidence": confidence,
                        "escalated": False,
                    }
                )

        # Default fallback
        return json.dumps(
            {
                "answer": "Thank you for contacting Zepto. Please provide more details about your query so we can assist you better.",
                "confidence": 0.5,
                "escalated": False,
            }
        )
