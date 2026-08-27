"""SkillForge AI — Vector Store.

Wraps FAISS for similarity search with namespace support.
Implementation planned for Milestone 2.
"""

from __future__ import annotations

from src.skillforge.models.resume import TextChunk


class VectorStore:
    """FAISS-based vector store with namespace support."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def upsert(self, namespace: str, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        """Add or update chunks in the specified namespace."""
        raise NotImplementedError("VectorStore will be implemented in Milestone 2")

    def search(self, query_embedding: list[float], namespace: str = "", top_k: int = 5) -> list[dict]:
        """Search for similar chunks by embedding."""
        raise NotImplementedError("VectorStore will be implemented in Milestone 2")

    def clear(self, namespace: str = "") -> None:
        """Clear all entries in a namespace."""
        raise NotImplementedError("VectorStore will be implemented in Milestone 2")
