"""
SkillForge AI — Embedding Engine.

Wraps SentenceTransformers to generate dense vector embeddings for
text. Embeddings are the foundation of semantic search, matching,
and the RAG pipeline — they capture *meaning* rather than keywords.

Design:
    - Lazy model loading: the model is only loaded on first use, not at import.
    - L2 normalization: embeddings are unit-length so cosine similarity = dot product.
    - Batch encoding: large text lists are encoded in configurable batches.
    - NumPy-native: all operations return numpy arrays for efficiency.

Usage:
    from src.skillforge.ai.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    emb = engine.encode_single("Senior Python developer")
    embs = engine.encode(["Python", "JavaScript", "React"])
    sim = engine.cosine_similarity(emb, embs[0])
"""

from __future__ import annotations

import numpy as np

from src.skillforge.utils.exceptions import EmbeddingError
from src.skillforge.utils.logging import logger


class EmbeddingEngine:
    """
    Generates sentence embeddings using SentenceTransformers.

    The model is loaded lazily on first encode() call and cached
    for all subsequent calls. Embeddings are L2-normalized so that
    cosine similarity can be computed as a simple dot product.

    Why embeddings matter (vs. keyword matching):
        "Python developer" and "Python programmer" share no exact words
        beyond "Python", but their embeddings will be very close (~0.95).
        "Team leadership" and "People management" share zero words, but
        their embeddings capture that they mean similar things (~0.75).
        This is what makes semantic matching genuinely AI-powered.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the embedding engine.

        Args:
            model_name: SentenceTransformer model identifier.
                        Default: all-MiniLM-L6-v2 (384-dim, fast, good quality).
            device: Torch device ('cpu', 'cuda', etc.). None = auto-detect.
            batch_size: Number of texts to encode per batch.
            normalize: Whether to L2-normalize embeddings. Enables dot-product
                       as cosine similarity.
        """
        self.model_name = model_name
        self._device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # Loaded lazily
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality (loads model if needed)."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into dense vector embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            numpy array of shape (len(texts), dimension).
            If normalize=True, each row is a unit vector.

        Raises:
            EmbeddingError: If encoding fails.
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        self._load_model()

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

            # Ensure 2D array even for single text
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            logger.debug(
                "Encoded texts",
                count=len(texts),
                shape=embeddings.shape,
            )

            return embeddings

        except Exception as e:
            logger.error("Encoding failed", error=str(e), count=len(texts))
            raise EmbeddingError(
                f"Failed to encode {len(texts)} texts: {e}",
                detail="The embedding model may have encountered an invalid input.",
            ) from e

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into a dense vector embedding.

        Args:
            text: String to encode.

        Returns:
            1-D numpy array of shape (dimension,).

        Raises:
            EmbeddingError: If encoding fails.
        """
        if not text or not text.strip():
            raise EmbeddingError(
                "Cannot encode empty text",
                detail="Provide a non-empty string to encode.",
            )

        embeddings = self.encode([text])
        return embeddings[0]

    # ── Similarity Methods ─────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        If both vectors are L2-normalized (which they are by default),
        this is equivalent to the dot product.

        Args:
            a: First embedding vector (1-D).
            b: Second embedding vector (1-D).

        Returns:
            Cosine similarity score in [-1, 1]. Higher = more similar.
        """
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def pairwise_cosine_similarity(
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between two sets of embeddings.

        Args:
            matrix_a: Embeddings of shape (M, D).
            matrix_b: Embeddings of shape (N, D).

        Returns:
            Similarity matrix of shape (M, N) where entry [i, j] is the
            cosine similarity between matrix_a[i] and matrix_b[j].
        """
        a = np.asarray(matrix_a, dtype=np.float32)
        b = np.asarray(matrix_b, dtype=np.float32)

        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        # Normalize rows
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)

        # Avoid division by zero
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)

        a_normalized = a / norm_a
        b_normalized = b / norm_b

        # Cosine similarity matrix via dot product
        return a_normalized @ b_normalized.T

    def similarity_between_texts(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two texts.

        Convenience method that encodes both texts and returns their
        cosine similarity.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity score in [-1, 1].
        """
        emb_a = self.encode_single(text_a)
        emb_b = self.encode_single(text_b)
        return self.cosine_similarity(emb_a, emb_b)

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """
        Find the most similar candidates to a query text.

        Args:
            query: The query text.
            candidates: List of candidate texts to compare against.
            top_k: Number of top results to return.

        Returns:
            List of (index, candidate_text, similarity_score) tuples,
            sorted by similarity descending.
        """
        if not candidates:
            return []

        query_emb = self.encode_single(query)
        candidate_embs = self.encode(candidates)

        similarities = self.pairwise_cosine_similarity(
            query_emb.reshape(1, -1), candidate_embs
        )[0]

        # Get top-k indices
        top_k = min(top_k, len(candidates))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (int(idx), candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]

    # ── Model Management ───────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the SentenceTransformer model (lazy, once)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model", model=self.model_name)

            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
            self._dimension = self._model.get_embedding_dimension()

            logger.info(
                "Embedding model loaded",
                model=self.model_name,
                dimension=self._dimension,
                device=str(self._model.device),
            )

        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is not installed",
                detail="Install with: pip install sentence-transformers",
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_name}': {e}",
                detail="Check that the model name is correct and you have internet access for first download.",
            ) from e

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory."""
        return self._model is not None

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        self._model = None
        self._dimension = None
        logger.info("Embedding model unloaded")
