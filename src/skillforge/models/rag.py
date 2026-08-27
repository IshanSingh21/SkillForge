"""
SkillForge AI — RAG (Retrieval-Augmented Generation) Data Models.

Pydantic models for the RAG pipeline: source citations, responses
with provenance, and conversation history management.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CitationSource(str, Enum):
    """Type of source document a citation refers to."""

    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"
    KNOWLEDGE_BASE = "knowledge_base"


class SourceCitation(BaseModel):
    """
    A single source citation attached to a RAG response.

    Links a claim or piece of information back to the specific
    chunk of text it was derived from.
    """

    source_type: CitationSource = Field(..., description="Type of source document")
    source_name: str = Field(default="", description="Document name or title")
    section: str = Field(default="", description="Section within the document")
    content_preview: str = Field(
        default="",
        description="Short preview of the cited content (first ~100 chars)",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score of this source to the query",
    )
    chunk_id: str = Field(default="", description="ID of the retrieved chunk")


class RAGResponse(BaseModel):
    """
    Complete response from the RAG career assistant.

    Contains the generated answer along with the source citations
    that ground the response in real data.
    """

    answer: str = Field(..., description="The generated response text")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations supporting the answer",
    )
    query: str = Field(default="", description="The original user query")
    model_used: str = Field(default="", description="Which LLM model generated this response")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated confidence in the response quality",
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of response generation",
    )

    @property
    def has_sources(self) -> bool:
        """Return True if the response has source citations."""
        return len(self.sources) > 0

    @property
    def source_summary(self) -> str:
        """Return a brief summary of sources used."""
        types = set(s.source_type.value for s in self.sources)
        return f"Sources: {', '.join(sorted(types))}" if types else "No sources"


class ConversationRole(str, Enum):
    """Role of a participant in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: ConversationRole = Field(..., description="Who sent this message")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was sent",
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations (for assistant messages)",
    )


class ConversationHistory(BaseModel):
    """Manages the conversation history for the RAG assistant."""

    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Ordered list of conversation messages",
    )
    max_messages: int = Field(
        default=20,
        description="Maximum messages to retain in history",
    )

    def add_message(self, role: ConversationRole, content: str, sources: list[SourceCitation] | None = None) -> None:
        """Add a message and trim history if over the limit."""
        message = ConversationMessage(
            role=role,
            content=content,
            sources=sources or [],
        )
        self.messages.append(message)

        # Keep only the most recent messages (always preserve the system message)
        if len(self.messages) > self.max_messages:
            system_msgs = [m for m in self.messages if m.role == ConversationRole.SYSTEM]
            non_system = [m for m in self.messages if m.role != ConversationRole.SYSTEM]
            self.messages = system_msgs + non_system[-(self.max_messages - len(system_msgs)):]

    def get_context_window(self, max_tokens_approx: int = 4000) -> list[ConversationMessage]:
        """Return recent messages that fit within an approximate token budget."""
        result: list[ConversationMessage] = []
        token_count = 0

        for msg in reversed(self.messages):
            # Rough approximation: 1 token ≈ 4 characters
            msg_tokens = len(msg.content) // 4
            if token_count + msg_tokens > max_tokens_approx:
                break
            result.append(msg)
            token_count += msg_tokens

        return list(reversed(result))

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
