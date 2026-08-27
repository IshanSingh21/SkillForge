"""SkillForge AI — RAG Career Assistant.

Retrieval-Augmented Generation pipeline for career Q&A.
Implementation planned for Milestone 4.
"""

from __future__ import annotations

from src.skillforge.models.rag import RAGResponse


class RAGAssistant:
    """RAG-based career assistant with source-aware responses."""

    def query(self, question: str, session_context: dict | None = None) -> RAGResponse:
        """Answer a career question using RAG."""
        raise NotImplementedError("RAGAssistant will be implemented in Milestone 4")
