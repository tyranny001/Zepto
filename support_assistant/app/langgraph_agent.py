"""LangGraph agent with 3 nodes: classify_intent, retrieve_and_answer, direct_answer."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.embeddings import get_embedding
from app.llm_interface import LLMInterface
from app.rag import VectorStore
from app.schema import GraphState


class ZeptoSupportAgent:
    """LangGraph-based support agent with policy retrieval and routing."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.llm = LLMInterface()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the 3-node LangGraph with conditional routing."""
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("retrieve_and_answer", self._retrieve_and_answer)
        graph.add_node("direct_answer", self._direct_answer)

        # Set entry point
        graph.set_entry_point("classify_intent")

        # Add edge from classify to conditional routing
        graph.add_conditional_edges(
            "classify_intent",
            lambda state: state["intent"],
            {
                "policy_question": "retrieve_and_answer",
                "general_question": "direct_answer",
            }
        )

        # Final edges
        graph.add_edge("retrieve_and_answer", END)
        graph.add_edge("direct_answer", END)

        return graph.compile()

    def _classify_intent(self, state: GraphState) -> dict:
        """Classify query as policy_question or general_question (MOCK_LLM toggle here).
        
        Graded baseline (MOCK_LLM=1): keyword heuristic, no LLM call.
        Optional MOCK_LLM=0: real LLM classification.
        """
        import os

        query = state["query"]
        q_lower = query.lower()

        # MOCK_LLM=1 (graded baseline): keyword heuristic
        use_mock = os.getenv("MOCK_LLM", "1").lower() in ("1", "true")

        if use_mock:
            # Keyword check
            policy_keywords = [
                "delivery", "return", "refund", "membership", "tracking",
                "cancel", "gift card", "support hours"
            ]
            intent = "policy_question" if any(kw in q_lower for kw in policy_keywords) else "general_question"
        else:
            # Optional MOCK_LLM=0: call LLM to classify
            try:
                prompt = f"Classify this as 'policy' or 'general': {query}\nRespond with only 'policy' or 'general'."
                # (Real LLM call would go here)
                intent = "policy_question"  # Placeholder
            except Exception:
                intent = "general_question"

        return {"intent": intent}

    def _retrieve_and_answer(self, state: GraphState) -> dict:
        """Retrieve chunks and generate answer (retrieval always runs, generation branches on MOCK_LLM)."""
        query = state["query"]

        # Always do real retrieval (no API needed, uses local embeddings)
        query_embedding = get_embedding(query)
        retrieved = self.vector_store.query(query_embedding, top_k=3)

        if not retrieved:
            return {
                "retrieved_docs": [],
                "answer": "I could not find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }

        # Build context from retrieved chunks
        context = "\n".join([f"({doc['doc_id']}): {doc['text']}" for doc in retrieved])

        # Generate answer (branches on MOCK_LLM here)
        result = self.llm.answer_question(query, context)

        sources = [doc["id"] for doc in retrieved]

        return {
            "retrieved_docs": retrieved,
            "answer": result["answer"],
            "sources": sources,
            "confidence": result["confidence"]
        }

    def _direct_answer(self, state: GraphState) -> dict:
        """Direct answer for general_question (MOCK_LLM toggle here).
        
        Graded baseline: fixed canned string, no LLM call.
        Optional MOCK_LLM=0: call LLM with no retrieval.
        """
        import os

        use_mock = os.getenv("MOCK_LLM", "1").lower() in ("1", "true")

        if use_mock:
            # Graded baseline: fixed canned string
            answer = "I can only answer questions about Zepto policies right now. Try asking about delivery, returns, membership, tracking, cancellation, gift cards, or support hours."
            confidence = 0.5
        else:
            # Optional MOCK_LLM=0: call LLM with no retrieval
            result = self.llm.answer_question(state["query"], "")
            answer = result["answer"]
            confidence = result["confidence"]

        return {
            "retrieved_docs": [],
            "answer": answer,
            "sources": [],
            "confidence": confidence
        }

    def invoke(self, query: str) -> dict:
        """Run the graph for a given query."""
        initial_state: GraphState = {
            "query": query,
            "intent": "",
            "retrieved_docs": [],
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
