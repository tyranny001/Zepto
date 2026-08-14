"""Pydantic schemas for requests and responses."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class SupportAnswer(BaseModel):
    """Response schema for /ask endpoint."""

    answer: str = Field(..., description="Zepto support agent answer (≤150 words)")
    sources: list[str] = Field(default_factory=list, description="Source FAQ sections")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score")
    escalated: bool = Field(default=False, description="Whether escalated to human agent")


class GraphState(TypedDict):
    """LangGraph state for support assistant."""

    question: str
    retrieved_chunks: list[str]
    draft_answer: str
    final_answer: str
    route: str  # "answer" | "escalate"
    confidence: float
    attempt_count: int
