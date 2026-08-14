"""Pydantic request and response schemas."""

from typing import TypedDict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")


class SupportResponse(BaseModel):
    """Response body from /ask endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: list[str] = Field(default_factory=list, description="List of chunk IDs used")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence 0-1")


class GraphState(TypedDict):
    """LangGraph state for the support assistant."""
    query: str
    intent: str  # "policy_question" or "general_question"
    retrieved_docs: list[dict]  # [{"id": str, "text": str, "doc_id": str}, ...]
    answer: str
    sources: list[str]
    confidence: float

