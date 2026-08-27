"""
SkillForge AI — Custom Exception Hierarchy.

Provides a structured exception hierarchy so that different error types
can be caught and handled appropriately at each layer of the application.
All custom exceptions inherit from SkillForgeError.

Usage:
    from src.skillforge.utils.exceptions import PDFParsingError

    try:
        text = parser.extract_text(pdf_bytes)
    except PDFParsingError as e:
        logger.error(f"Failed to parse PDF: {e}")
"""

from __future__ import annotations


class SkillForgeError(Exception):
    """Base exception for all SkillForge AI errors."""

    def __init__(self, message: str = "", detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


# ── Data Layer Exceptions ──────────────────────────────────────────────


class PDFParsingError(SkillForgeError):
    """Raised when PDF text extraction fails."""

    pass


class PreprocessingError(SkillForgeError):
    """Raised when text cleaning or section extraction fails."""

    pass


class ChunkingError(SkillForgeError):
    """Raised when document chunking fails."""

    pass


# ── AI Layer Exceptions ───────────────────────────────────────────────


class EmbeddingError(SkillForgeError):
    """Raised when embedding generation fails (model load, encoding)."""

    pass


class VectorStoreError(SkillForgeError):
    """Raised when vector store operations fail (indexing, search)."""

    pass


class LLMError(SkillForgeError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when the LLM API is unreachable."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message)


class LLMResponseError(LLMError):
    """Raised when the LLM returns an invalid or unparseable response."""

    pass


# ── Service Layer Exceptions ──────────────────────────────────────────


class SkillExtractionError(SkillForgeError):
    """Raised when skill extraction fails."""

    pass


class MatchingError(SkillForgeError):
    """Raised when semantic matching fails."""

    pass


class RAGError(SkillForgeError):
    """Raised when the RAG pipeline fails."""

    pass


class RoadmapGenerationError(SkillForgeError):
    """Raised when learning roadmap generation fails."""

    pass


# ── Configuration Exceptions ──────────────────────────────────────────


class ConfigurationError(SkillForgeError):
    """Raised when required configuration is missing or invalid."""

    pass
