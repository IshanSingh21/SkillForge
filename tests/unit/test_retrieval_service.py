"""
Tests for src.skillforge.services.retrieval_service — Vector Retrieval Service.

Tests cover:
- Document chunking & embedding generation
- Resume indexing & targeted retrieval
- Job description indexing & targeted retrieval
- Knowledge base markdown document indexing & retrieval
- Multi-source combined retrieval and SourceCitation models
- Relevance score filtering and top-k limits
- Chunk size and overlap integrity
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.ai.vector_store import VectorStore
from src.skillforge.data.chunker import TextChunker
from src.skillforge.models.rag import CitationSource, SourceCitation
from src.skillforge.models.resume import ResumeAnalysis, ResumeSection
from src.skillforge.services.retrieval_service import RetrievalService


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def retrieval_service(engine: EmbeddingEngine) -> RetrievalService:
    """Fresh RetrievalService instance with injected components."""
    store = VectorStore(dimension=engine.dimension)
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    return RetrievalService(vector_store=store, embedding_engine=engine, chunker=chunker)


# ── Sample Data ────────────────────────────────────────────────────────

SAMPLE_RESUME = """
Jane Doe — Senior Backend Developer
EXPERIENCE
Senior Software Engineer at TechCorp (2021-Present)
- Built distributed payment microservices in Python using FastAPI, PostgreSQL, and Redis.
- Implemented asynchronous Celery task queues handling 100k events/hour.
- Deployed infrastructure on AWS (EKS, RDS, S3) with Terraform.

Software Engineer at WebWorks (2018-2021)
- Developed RESTful APIs and optimized database queries in PostgreSQL.
- Maintained Docker containers and CI/CD pipelines in GitHub Actions.
"""

SAMPLE_JD = """
Staff Backend Engineer — FinTech Innovations
Requirements:
- 5+ years of experience designing scalable microservices in Python or Go.
- Deep expertise with relational databases (PostgreSQL) and caching (Redis).
- Hands-on experience with Kubernetes, Docker, and AWS cloud environments.
- Strong understanding of event-driven architectures (Kafka, RabbitMQ).
"""


# ── Ingestion & Indexing Tests ─────────────────────────────────────────


class TestRetrievalIndexing:
    """Tests for chunking, embedding, and indexing documents."""

    def test_index_resume_text(self, retrieval_service: RetrievalService):
        """Should chunk, embed, and index resume text."""
        count = retrieval_service.index_resume(SAMPLE_RESUME, filename="jane_resume.pdf")

        assert count > 0
        assert retrieval_service.vector_store.count("resume") == count

    def test_index_resume_analysis_object(self, retrieval_service: RetrievalService):
        """Should index pre-chunked ResumeAnalysis objects directly."""
        analysis = ResumeAnalysis(
            raw_text=SAMPLE_RESUME,
            cleaned_text=SAMPLE_RESUME,
            sections=[
                ResumeSection(title="Experience", content="Built payment microservices in Python."),
            ],
            chunks=[
                retrieval_service.chunker.chunk_text(
                    SAMPLE_RESUME, source="resume", source_name="jane_resume.pdf"
                )[0]
            ],
        )

        count = retrieval_service.index_resume(analysis)
        assert count == 1
        assert retrieval_service.vector_store.count("resume") == 1

    def test_index_job_description(self, retrieval_service: RetrievalService):
        """Should index job description into 'job_description' namespace."""
        count = retrieval_service.index_job_description(SAMPLE_JD, title="Staff Backend Engineer")

        assert count > 0
        assert retrieval_service.vector_store.count("job_description") == count

    def test_index_knowledge_base_directory(self, retrieval_service: RetrievalService):
        """Should index all markdown files from the knowledge_base folder."""
        count = retrieval_service.index_knowledge_base_directory()

        assert count > 0
        assert retrieval_service.vector_store.count("knowledge_base") == count
        assert retrieval_service.vector_store.count("knowledge_base") >= 4  # At least 4 docs


# ── Retrieval Tests ────────────────────────────────────────────────────


class TestSimilarityRetrieval:
    """Tests for multi-source similarity retrieval and citation generation."""

    @pytest.fixture(autouse=True)
    def populate_index(self, retrieval_service: RetrievalService):
        """Populate store with resume, JD, and knowledge base."""
        retrieval_service.index_resume(SAMPLE_RESUME, filename="jane_resume.pdf")
        retrieval_service.index_job_description(SAMPLE_JD, title="Staff Backend Engineer")
        retrieval_service.index_knowledge_base_directory()

    def test_retrieve_from_resume(self, retrieval_service: RetrievalService):
        """Should retrieve citations specifically from the candidate's resume."""
        citations = retrieval_service.retrieve_from_resume(
            query="What experience does the candidate have with AWS and Terraform?",
            top_k=2,
        )

        assert len(citations) > 0
        assert all(c.source_type == CitationSource.RESUME for c in citations)
        assert citations[0].relevance_score > 0.40
        assert any("AWS" in c.content_preview for c in citations)

    def test_retrieve_from_job_description(self, retrieval_service: RetrievalService):
        """Should retrieve citations specifically from the job description."""
        citations = retrieval_service.retrieve_from_job_description(
            query="What are the database and caching requirements?",
            top_k=2,
        )

        assert len(citations) > 0
        assert all(c.source_type == CitationSource.JOB_DESCRIPTION for c in citations)
        assert any("PostgreSQL" in c.content_preview or "Redis" in c.content_preview for c in citations)

    def test_retrieve_from_knowledge_base(self, retrieval_service: RetrievalService):
        """Should retrieve citations specifically from the knowledge base."""
        citations = retrieval_service.retrieve_from_knowledge_base(
            query="How should I structure behavioral interview responses using the STAR method?",
            top_k=2,
        )

        assert len(citations) > 0
        assert all(c.source_type == CitationSource.KNOWLEDGE_BASE for c in citations)
        assert any("STAR" in c.content_preview for c in citations)

    def test_retrieve_multi_source_combined(self, retrieval_service: RetrievalService):
        """Should retrieve citations across multiple sources and sort by relevance."""
        citations = retrieval_service.retrieve(
            query="Python microservices and AWS infrastructure",
            sources=[CitationSource.RESUME, CitationSource.JOB_DESCRIPTION],
            top_k=4,
        )

        assert len(citations) >= 2
        sources = {c.source_type for c in citations}
        assert CitationSource.RESUME in sources or CitationSource.JOB_DESCRIPTION in sources

        # Relevance scores should be descending
        scores = [c.relevance_score for c in citations]
        assert scores == sorted(scores, reverse=True)

    def test_source_citation_attributes(self, retrieval_service: RetrievalService):
        """SourceCitation objects should have all required metadata attributes populated."""
        citations = retrieval_service.retrieve("FastAPI PostgreSQL", top_k=1)

        assert len(citations) == 1
        c = citations[0]
        assert isinstance(c, SourceCitation)
        assert isinstance(c.source_type, CitationSource)
        assert len(c.source_name) > 0
        assert len(c.content_preview) > 0
        assert 0.0 <= c.relevance_score <= 1.0
        assert len(c.chunk_id) > 0


# ── Chunking Integrity Tests ───────────────────────────────────────────


class TestChunkingIntegrity:
    """Tests verifying chunk size and overlap choices."""

    def test_chunk_size_and_overlap_parameters(self, retrieval_service: RetrievalService):
        """RetrievalService chunker should have standard chunk size (512) and overlap (50)."""
        assert retrieval_service.chunker.chunk_size == 512
        assert retrieval_service.chunker.chunk_overlap == 50

    def test_chunks_respect_chunk_size(self, retrieval_service: RetrievalService):
        """Generated chunks should not exceed chunk_size + overlap overhead."""
        long_text = "This is a sentence about backend engineering. " * 30
        chunks = retrieval_service.chunker.chunk_text(long_text, source="resume")

        for c in chunks:
            assert len(c.content) <= 512 + 60  # Allows slight margin for overlap prefix


# ── Status and Edge Cases Tests ────────────────────────────────────────


class TestRetrievalEdgeCases:
    """Tests for edge cases and status reporting."""

    def test_get_status_reports_correct_counts(self, retrieval_service: RetrievalService):
        """get_status should accurately report namespace counts."""
        retrieval_service.index_resume(SAMPLE_RESUME)
        status = retrieval_service.get_status()

        assert status["total_chunks"] > 0
        assert "resume" in status["namespaces"]
        assert status["dimension"] == 384

    def test_empty_query_returns_empty_citations(self, retrieval_service: RetrievalService):
        """Empty query should return empty list."""
        assert retrieval_service.retrieve("") == []
        assert retrieval_service.retrieve("   ") == []

    def test_clear_resets_store(self, retrieval_service: RetrievalService):
        """Clear should remove all indexed chunks."""
        retrieval_service.index_resume(SAMPLE_RESUME)
        assert retrieval_service.vector_store.count("resume") > 0

        retrieval_service.clear("resume")
        assert retrieval_service.vector_store.count("resume") == 0
