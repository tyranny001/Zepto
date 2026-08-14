"""LangGraph state machine: retrieve → generate → validate."""

from __future__ import annotations

import json
import re
from typing import Any

from langgraph.graph import END, StateGraph

from app.llm import LLMInterface
from app.rag import RAGSystem
from app.schema import GraphState, SupportAnswer


def create_graph(rag_system: RAGSystem) -> StateGraph:
    """Build LangGraph with retrieve → generate → validate flow."""

    llm = LLMInterface()

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        """Retrieve relevant chunks from corpus."""
        question = state["question"]
        chunks = rag_system.retrieve(question, top_k=3)
        return {
            "retrieved_chunks": chunks,
            "draft_answer": "",
            "attempt_count": 0,
        }

    def generate_node(state: GraphState) -> dict[str, Any]:
        """Generate answer using LLM + retrieved context."""
        question = state["question"]
        context = "\n".join(state["retrieved_chunks"])

        system_prompt = """You are a helpful Zepto grocery customer support agent.
Use the provided FAQ context to answer questions accurately.
IMPORTANT: Do NOT invent refund policies, delivery terms, or discounts not in the context.
If you don't know, escalate the question.
Respond in valid JSON with keys: answer (str, ≤150 words), confidence (float 0-1), escalated (bool)."""

        user_prompt = f"""FAQ Context:
{context}

Customer Question: {question}

Provide a helpful response in JSON format."""

        response_text = llm.call(user_prompt, system=system_prompt, max_tokens=400)

        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            answer = response_json.get("answer", "")
            confidence = float(response_json.get("confidence", 0.5))
            escalated = bool(response_json.get("escalated", False))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: extract answer from plain text
            answer = response_text[:150]
            confidence = 0.3
            escalated = True

        return {
            "draft_answer": answer,
            "confidence": confidence,
            "route": "escalate" if escalated else "answer",
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    def validate_node(state: GraphState) -> dict[str, Any]:
        """Validate answer; route to END or retry generate."""
        confidence = state["confidence"]
        answer = state["draft_answer"]
        route = state.get("route", "answer")
        attempt_count = state.get("attempt_count", 1)

        # Validation: confidence threshold and answer quality
        is_confident = confidence >= 0.6
        is_non_empty = bool(answer and len(answer) > 10)

        # Route logic
        if is_confident and is_non_empty:
            route = "answer"
        elif attempt_count < 2 and not is_confident:
            route = "regenerate"
        else:
            route = "escalate"

        return {"final_answer": answer, "route": route}

    def should_continue(state: GraphState) -> str:
        """Routing logic after validate node."""
        route = state.get("route", "answer")
        if route == "regenerate":
            return "generate"
        return END

    # Build graph
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", should_continue, {"generate": "generate", END: END})

    graph.set_entry_point("retrieve")

    return graph.compile()
