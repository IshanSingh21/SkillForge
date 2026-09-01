"""
Tests for src.skillforge.ai.vector_store — FAISS Vector Store.

Tests cover:
- Ingestion and chunk storage in FAISS
- Partitioned namespace isolation ('resume', 'job_description', 'knowledge_base')
- Similarity search with top-k and min_score thresholds
- Cross-namespace multi-source retrieval
- Metadata filtering (by section, custom attributes)
- Chunk lookup by ID
- Index persistence (saving to disk and reloading)
- Dimension mismatch and error handling
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.ai.vector_store import SearchResult, VectorStore
from src.skillforge.models.resume import TextChunk
from src.skillforge.utils.exceptions import VectorStoreError


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine for vector store tests."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(engine: EmbeddingEngine) -> VectorStore:
    """Return a fresh VectorStore instance."""
    return VectorStore(dimension=engine.dimension)


def _make_chunk(chunk_id: str, content: str, source: str = "resume", section: str = "") -> TextChunk:
    """Helper to construct TextChunk."""
    return TextChunk(
        chunk_id=chunk_id,
        content=content,
        source=source,
        source_name=f"{source}_doc.txt",
        section=section,
        chunk_index=0,
    )


# ── Ingestion & Namespace Tests ────────────────────────────────────────


class TestVectorStoreIngestion:
    """Tests for adding chunks and managing namespaces."""

    def test_add_chunks_to_namespace(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Should add chunks and vectors into a specific namespace."""
        chunks = [
            _make_chunk("c1", "Python backend engineering with FastAPI", source="resume"),
            _make_chunk("c2", "PostgreSQL database optimization and caching", source="resume"),
        ]
        embeddings = engine.encode([c.content for c in chunks])

        added_ids = vector_store.add_chunks(chunks, embeddings, namespace="resume")

        assert len(added_ids) == 2
        assert vector_store.count("resume") == 2
        assert vector_store.count() == 2

    def test_multiple_namespaces_are_isolated(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Different namespaces should store distinct vectors."""
        resume_chunks = [_make_chunk("r1", "Jane Doe resume details", source="resume")]
        jd_chunks = [_make_chunk("jd1", "Job requirements for backend engineer", source="job_description")]

        vector_store.add_chunks(resume_chunks, engine.encode(["Jane Doe resume details"]), namespace="resume")
        vector_store.add_chunks(jd_chunks, engine.encode(["Job requirements for backend engineer"]), namespace="job_description")

        assert vector_store.count("resume") == 1
        assert vector_store.count("job_description") == 1
        assert vector_store.count() == 2
        assert set(vector_store.list_namespaces()) == {"resume", "job_description"}

    def test_upsert_replaces_existing_namespace_content(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Upsert should clear the old namespace data before inserting."""
        chunks1 = [_make_chunk("old1", "Old resume content", source="resume")]
        vector_store.add_chunks(chunks1, engine.encode(["Old resume content"]), namespace="resume")
        assert vector_store.count("resume") == 1

        chunks2 = [
            _make_chunk("new1", "New resume section 1", source="resume"),
            _make_chunk("new2", "New resume section 2", source="resume"),
        ]
        vector_store.upsert(namespace="resume", chunks=chunks2, embeddings=engine.encode([c.content for c in chunks2]))

        assert vector_store.count("resume") == 2
        assert vector_store.get_chunk("old1", namespace="resume") is None
        assert vector_store.get_chunk("new1", namespace="resume") is not None


# ── Search & Similarity Retrieval Tests ────────────────────────────────


class TestVectorStoreSearch:
    """Tests for vector similarity retrieval."""

    @pytest.fixture(autouse=True)
    def setup_populated_store(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Populate store with sample documents across 3 namespaces."""
        resume_chunks = [
            _make_chunk("r1", "Designed scalable microservices using Python and FastAPI on AWS EC2.", source="resume", section="Experience"),
            _make_chunk("r2", "Master of Science in Computer Science from MIT.", source="resume", section="Education"),
        ]
        jd_chunks = [
            _make_chunk("jd1", "Seeking senior engineer with 5+ years Python and cloud microservices experience.", source="job_description", section="Requirements"),
            _make_chunk("jd2", "Competitive salary, 401k matching, and health benefits.", source="job_description", section="Benefits"),
        ]
        kb_chunks = [
            _make_chunk("kb1", "Use the STAR method (Situation, Task, Action, Result) for behavioral interview questions.", source="knowledge_base", section="Interview Tips"),
            _make_chunk("kb2", "When designing distributed systems, consider the CAP theorem trade-offs.", source="knowledge_base", section="System Design"),
        ]

        vector_store.add_chunks(resume_chunks, engine.encode([c.content for c in resume_chunks]), namespace="resume")
        vector_store.add_chunks(jd_chunks, engine.encode([c.content for c in jd_chunks]), namespace="job_description")
        vector_store.add_chunks(kb_chunks, engine.encode([c.content for c in kb_chunks]), namespace="knowledge_base")

    def test_search_single_namespace(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Searching a specific namespace should only return results from that partition."""
        q_emb = engine.encode_single("Python FastAPI microservices")
        results = vector_store.search(q_emb, namespace="resume", top_k=2)

        assert len(results) > 0
        assert all(r.chunk.source == "resume" for r in results)
        assert results[0].chunk.chunk_id == "r1"
        assert results[0].score > 0.65

    def test_search_across_all_namespaces(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Searching with namespace='' should search all partitions and rank globally."""
        q_emb = engine.encode_single("Python cloud microservices")
        results = vector_store.search(q_emb, namespace="", top_k=4)

        assert len(results) >= 2
        sources_found = {r.chunk.source for r in results}
        # Both r1 and jd1 talk about Python microservices
        assert "resume" in sources_found or "job_description" in sources_found

        # Scores should be in strictly descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_by_text_convenience(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """search_by_text should encode query and return top results."""
        results = vector_store.search_by_text(
            query="STAR method behavioral interview",
            embedding_engine=engine,
            namespace="knowledge_base",
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "kb1"
        assert "STAR method" in results[0].chunk.content

    def test_metadata_filtering(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Should filter search results by section or metadata attribute."""
        q_emb = engine.encode_single("Academic degree and university")
        results = vector_store.search(
            q_emb,
            namespace="resume",
            filter_metadata={"section": "Education"},
            top_k=2,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "r2"
        assert results[0].chunk.section == "Education"

    def test_min_score_filtering(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """Results below min_score threshold should be excluded."""
        q_emb = engine.encode_single("Completely irrelevant query about baking chocolate cookies")
        results = vector_store.search(q_emb, namespace="", min_score=0.75)

        assert len(results) == 0


# ── Persistence & Lookup Tests ─────────────────────────────────────────


class TestVectorStorePersistence:
    """Tests for saving to and loading from disk."""

    def test_save_and_load_roundtrip(self, vector_store: VectorStore, engine: EmbeddingEngine):
        """FAISS indices and metadata should persist accurately to disk."""
        chunks = [
            _make_chunk("p1", "First persistent chunk", source="resume", section="Exp"),
            _make_chunk("p2", "Second persistent chunk", source="knowledge_base", section="Guide"),
        ]
        vector_store.add_chunks([chunks[0]], engine.encode([chunks[0].content]), namespace="resume")
        vector_store.add_chunks([chunks[1]], engine.encode([chunks[1].content]), namespace="knowledge_base")

        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store.save(temp_dir)

            # Load into a new VectorStore instance
            loaded_store = VectorStore(dimension=engine.dimension)
            loaded_store.load(temp_dir)

            assert loaded_store.count("resume") == 1
            assert loaded_store.count("knowledge_base") == 1
            assert loaded_store.count() == 2

            # Verify chunk content and metadata
            retrieved = loaded_store.get_chunk("p1", namespace="resume")
            assert retrieved is not None
            assert retrieved.content == "First persistent chunk"
            assert retrieved.section == "Exp"

            # Search loaded store
            q_emb = engine.encode_single("persistent chunk")
            results = loaded_store.search(q_emb, top_k=2)
            assert len(results) == 2


# ── Error Handling Tests ───────────────────────────────────────────────


class TestVectorStoreErrors:
    """Tests for edge cases and errors."""

    def test_dimension_mismatch_raises_error(self, vector_store: VectorStore):
        """Adding embeddings with incorrect dimensions should raise VectorStoreError."""
        chunks = [_make_chunk("err1", "Test content")]
        wrong_embs = np.zeros((1, 128), dtype=np.float32)  # Store expects 384

        with pytest.raises(VectorStoreError, match="dimension mismatch"):
            vector_store.add_chunks(chunks, wrong_embs, namespace="resume")

    def test_count_mismatch_raises_error(self, vector_store: VectorStore):
        """Mismatch between chunks list and embeddings array size should raise error."""
        chunks = [_make_chunk("e1", "c1"), _make_chunk("e2", "c2")]
        embs = np.zeros((1, vector_store.dimension), dtype=np.float32)

        with pytest.raises(VectorStoreError, match="Count mismatch"):
            vector_store.add_chunks(chunks, embs, namespace="resume")

    def test_load_nonexistent_manifest_raises_error(self, vector_store: VectorStore):
        """Loading from an invalid directory should raise VectorStoreError."""
        with pytest.raises(VectorStoreError, match="manifest not found"):
            vector_store.load("/nonexistent/directory/path/12345")
