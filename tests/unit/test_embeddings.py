"""
Tests for src.skillforge.ai.embeddings — Embedding Engine.

Tests cover:
- Single and batch encoding
- Embedding dimensions and shapes
- L2 normalization
- Cosine similarity (single pair + pairwise matrix)
- Semantic similarity properties (similar texts → high score)
- find_most_similar utility
- Edge cases (empty input, model loading)

Note: These tests load the actual SentenceTransformer model, so the
first run downloads ~80MB. Subsequent runs use the cached model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.utils.exceptions import EmbeddingError


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """
    Create a shared EmbeddingEngine for all tests in this module.

    scope="module" ensures the model is loaded only once per test run,
    not once per test function — critical for speed.
    """
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


# ── Encoding Tests ─────────────────────────────────────────────────────


class TestEncoding:
    """Tests for text encoding."""

    def test_encode_single_returns_1d(self, engine: EmbeddingEngine):
        """Single text encoding should return a 1-D vector."""
        emb = engine.encode_single("Python developer")
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.shape[0] == engine.dimension

    def test_encode_batch_returns_2d(self, engine: EmbeddingEngine):
        """Batch encoding should return a 2-D matrix."""
        texts = ["Python", "JavaScript", "React"]
        embs = engine.encode(texts)

        assert isinstance(embs, np.ndarray)
        assert embs.ndim == 2
        assert embs.shape == (3, engine.dimension)

    def test_encode_empty_list_returns_empty_array(self, engine: EmbeddingEngine):
        """Empty input should return an empty array with correct dimensions."""
        embs = engine.encode([])
        assert embs.shape == (0, engine.dimension)

    def test_encode_single_rejects_empty(self, engine: EmbeddingEngine):
        """Empty string should raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty"):
            engine.encode_single("")

    def test_dimension_is_384(self, engine: EmbeddingEngine):
        """all-MiniLM-L6-v2 should produce 384-dimensional embeddings."""
        assert engine.dimension == 384


class TestNormalization:
    """Tests for L2 normalization."""

    def test_embeddings_are_unit_length(self, engine: EmbeddingEngine):
        """With normalize=True, each embedding should have L2 norm ≈ 1.0."""
        embs = engine.encode(["Python", "JavaScript", "Machine Learning"])

        for i in range(len(embs)):
            norm = np.linalg.norm(embs[i])
            assert abs(norm - 1.0) < 1e-5, f"Row {i} norm = {norm}, expected ~1.0"

    def test_normalized_dot_product_equals_cosine(self, engine: EmbeddingEngine):
        """For normalized vectors, dot product should equal cosine similarity."""
        emb_a = engine.encode_single("Python developer")
        emb_b = engine.encode_single("Python programmer")

        dot = float(np.dot(emb_a, emb_b))
        cosine = engine.cosine_similarity(emb_a, emb_b)

        assert abs(dot - cosine) < 1e-5


# ── Cosine Similarity Tests ───────────────────────────────────────────


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_texts_have_similarity_1(self, engine: EmbeddingEngine):
        """The same text encoded twice should have similarity ≈ 1.0."""
        emb = engine.encode_single("Senior Python developer")
        sim = engine.cosine_similarity(emb, emb)
        assert sim > 0.999

    def test_similar_texts_have_high_similarity(self, engine: EmbeddingEngine):
        """Semantically similar texts should have high similarity."""
        emb_a = engine.encode_single("Python developer")
        emb_b = engine.encode_single("Python programmer")
        sim = engine.cosine_similarity(emb_a, emb_b)

        assert sim > 0.7, f"Expected >0.7 for similar texts, got {sim:.3f}"

    def test_unrelated_texts_have_low_similarity(self, engine: EmbeddingEngine):
        """Completely unrelated texts should have low similarity."""
        emb_a = engine.encode_single("Python machine learning engineer")
        emb_b = engine.encode_single("Italian pasta recipe with tomato sauce")
        sim = engine.cosine_similarity(emb_a, emb_b)

        assert sim < 0.3, f"Expected <0.3 for unrelated texts, got {sim:.3f}"

    def test_synonym_skills_are_similar(self, engine: EmbeddingEngine):
        """Skills that are synonyms should have high embedding similarity."""
        pairs = [
            ("team leadership", "people management"),
            ("Amazon Web Services", "AWS cloud"),
            ("machine learning", "ML algorithms"),
        ]
        for text_a, text_b in pairs:
            sim = engine.similarity_between_texts(text_a, text_b)
            assert sim > 0.40, (
                f"Expected synonym similarity >0.40: "
                f"'{text_a}' vs '{text_b}' = {sim:.3f}"
            )

    def test_zero_vector_similarity(self, engine: EmbeddingEngine):
        """Similarity with a zero vector should return 0."""
        emb = engine.encode_single("Python")
        zero = np.zeros_like(emb)
        assert engine.cosine_similarity(emb, zero) == 0.0


class TestPairwiseSimilarity:
    """Tests for pairwise cosine similarity matrix."""

    def test_pairwise_shape(self, engine: EmbeddingEngine):
        """Pairwise matrix should have shape (M, N)."""
        a = engine.encode(["Python", "Java"])
        b = engine.encode(["JavaScript", "TypeScript", "React"])

        sim_matrix = engine.pairwise_cosine_similarity(a, b)
        assert sim_matrix.shape == (2, 3)

    def test_pairwise_diagonal_identity(self, engine: EmbeddingEngine):
        """Similarity of identical embeddings should be ~1 on the diagonal."""
        texts = ["Python", "JavaScript", "Docker"]
        embs = engine.encode(texts)

        sim_matrix = engine.pairwise_cosine_similarity(embs, embs)

        for i in range(len(texts)):
            assert sim_matrix[i, i] > 0.999

    def test_pairwise_symmetry(self, engine: EmbeddingEngine):
        """sim(a, b) should approximately equal sim(b, a)."""
        a = engine.encode(["Python", "Docker"])
        b = engine.encode(["JavaScript", "Kubernetes"])

        sim_ab = engine.pairwise_cosine_similarity(a, b)
        sim_ba = engine.pairwise_cosine_similarity(b, a)

        # sim_ab[i,j] should ≈ sim_ba[j,i]
        np.testing.assert_array_almost_equal(sim_ab, sim_ba.T, decimal=5)

    def test_pairwise_values_in_range(self, engine: EmbeddingEngine):
        """All similarity values should be in [-1, 1]."""
        a = engine.encode(["Python", "Leadership", "AWS"])
        b = engine.encode(["Java", "Mentoring", "Azure", "Docker"])

        sim = engine.pairwise_cosine_similarity(a, b)
        assert np.all(sim >= -1.0 - 1e-6) and np.all(sim <= 1.0 + 1e-6)


# ── find_most_similar Tests ───────────────────────────────────────────


class TestFindMostSimilar:
    """Tests for the find_most_similar utility."""

    def test_returns_correct_top_k(self, engine: EmbeddingEngine):
        """Should return exactly top_k results."""
        candidates = ["Python", "JavaScript", "Docker", "Kubernetes", "AWS"]
        results = engine.find_most_similar("Python programming", candidates, top_k=3)

        assert len(results) == 3

    def test_results_are_sorted_descending(self, engine: EmbeddingEngine):
        """Results should be sorted by similarity, highest first."""
        candidates = ["Python", "JavaScript", "Italian cooking", "Docker"]
        results = engine.find_most_similar("Python developer", candidates, top_k=4)

        scores = [score for _, _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_is_most_similar(self, engine: EmbeddingEngine):
        """The top result should be the most semantically similar candidate."""
        candidates = ["Italian pasta", "Python programming", "Weather forecast"]
        results = engine.find_most_similar("Python developer", candidates, top_k=1)

        assert len(results) == 1
        assert results[0][1] == "Python programming"

    def test_empty_candidates(self, engine: EmbeddingEngine):
        """Empty candidate list should return empty results."""
        results = engine.find_most_similar("Python", [], top_k=3)
        assert results == []


# ── Model Management Tests ─────────────────────────────────────────────


class TestModelManagement:
    """Tests for lazy loading and model lifecycle."""

    def test_model_not_loaded_at_init(self):
        """Model should not load until first encode call."""
        eng = EmbeddingEngine.__new__(EmbeddingEngine)
        eng.model_name = "all-MiniLM-L6-v2"
        eng._device = None
        eng.batch_size = 32
        eng.normalize = True
        eng._model = None
        eng._dimension = None
        assert not eng.is_loaded

    def test_model_loads_on_first_encode(self, engine: EmbeddingEngine):
        """After encoding, model should be loaded."""
        engine.encode_single("test")
        assert engine.is_loaded
