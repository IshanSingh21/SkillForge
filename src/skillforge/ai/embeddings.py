"""SkillForge AI — Embedding Engine.

Wraps SentenceTransformers for generating dense vector embeddings.
Implementation planned for Milestone 2.
"""

from __future__ import annotations


class EmbeddingEngine:
    """Generates sentence embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        # Model will be loaded lazily in Milestone 2
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into dense vector embeddings."""
        raise NotImplementedError("EmbeddingEngine will be implemented in Milestone 2")

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text into a dense vector embedding."""
        raise NotImplementedError("EmbeddingEngine will be implemented in Milestone 2")
