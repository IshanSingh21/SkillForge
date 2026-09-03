"""
Tests for Milestone 8 — Curated Career Knowledge Base & RAG Integration.

Tests cover:
- KnowledgeBaseLoader document loading, title extraction, and section parsing
- Preservation of metadata (topic, doc_id, section, file_path, tags)
- Chunking with section-aware TextChunker
- Ingestion into FAISS 'knowledge_base' vector store partition
- Semantic retrieval across all 10 core domain areas (ML, DL, NLP, CV, SQL, Python, MLOps, Cloud, DSA, GenAI)
- Topic-filtered career knowledge retrieval
- Integration with RAG pipeline and career recommendation generation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.ai.llm.mock import MockLLMProvider
from src.skillforge.ai.vector_store import VectorStore
from src.skillforge.data.chunker import TextChunker
from src.skillforge.data.knowledge_base_loader import DocumentSection, KnowledgeBaseLoader, KnowledgeDocument
from src.skillforge.models.rag import CitationSource, SourceCitation
from src.skillforge.services.rag_assistant import RAGAssistant
from src.skillforge.services.retrieval_service import RetrievalService

KB_DIR = PROJECT_ROOT / "knowledge_base"

# The 10 required domain topics
REQUIRED_TOPIC_FILES = [
    "machine_learning.md",
    "deep_learning.md",
    "natural_language_processing.md",
    "computer_vision.md",
    "sql_and_databases.md",
    "python_development.md",
    "mlops.md",
    "cloud_computing.md",
    "data_structures_and_algorithms.md",
    "generative_ai.md",
]


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def kb_loader() -> KnowledgeBaseLoader:
    """KnowledgeBaseLoader instance."""
    return KnowledgeBaseLoader()


@pytest.fixture
def chunker() -> TextChunker:
    """TextChunker instance."""
    return TextChunker(chunk_size=512, chunk_overlap=50)


@pytest.fixture
def indexed_retrieval_service(engine: EmbeddingEngine) -> RetrievalService:
    """RetrievalService with all knowledge base documents indexed."""
    store = VectorStore(dimension=engine.dimension)
    service = RetrievalService(vector_store=store, embedding_engine=engine)
    service.index_knowledge_base_directory(KB_DIR)
    return service


# ── Document Loading & Parsing Tests ───────────────────────────────────


class TestKnowledgeBaseLoader:
    """Tests for loading and parsing markdown knowledge base files."""

    def test_all_10_required_topics_exist(self):
        """All 10 required topic markdown files must exist in knowledge_base directory."""
        for filename in REQUIRED_TOPIC_FILES:
            file_path = KB_DIR / filename
            assert file_path.exists(), f"Missing required knowledge file: {filename}"
            assert file_path.stat().st_size > 300, f"Knowledge file too small: {filename}"

    def test_load_single_document_structure(self, kb_loader: KnowledgeBaseLoader):
        """Should parse H1 title, sections, and metadata from a markdown document."""
        ml_path = KB_DIR / "machine_learning.md"
        doc = kb_loader.load_file(ml_path)

        assert isinstance(doc, KnowledgeDocument)
        assert doc.doc_id == "machine_learning"
        assert "Machine Learning" in doc.title
        assert doc.topic == "Machine Learning"
        assert len(doc.sections) >= 5
        assert len(doc.tags) > 0

        # Check section headings
        headings = [s.heading for s in doc.sections]
        assert any("Fundamental Concepts" in h for h in headings)
        assert any("Core Tools" in h or "Libraries" in h for h in headings)
        assert any("Career Roles" in h for h in headings)
        assert any("Learning Progression" in h for h in headings)

    def test_load_all_documents_in_directory(self, kb_loader: KnowledgeBaseLoader):
        """Should load all markdown files in the knowledge_base directory."""
        docs = kb_loader.load_directory(KB_DIR)

        # 10 topic documents + 4 general career guide documents = 14 total
        assert len(docs) >= 14
        doc_ids = {d.doc_id for d in docs}
        assert "machine_learning" in doc_ids
        assert "deep_learning" in doc_ids
        assert "natural_language_processing" in doc_ids
        assert "generative_ai" in doc_ids
        assert "mlops" in doc_ids

    def test_chunk_document_preserves_section_and_topic_metadata(
        self,
        kb_loader: KnowledgeBaseLoader,
        chunker: TextChunker,
    ):
        """Chunks should inherit section name, topic, doc_id, and source."""
        doc = kb_loader.load_file(KB_DIR / "generative_ai.md")
        chunks = kb_loader.chunk_document(doc, chunker)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.source == CitationSource.KNOWLEDGE_BASE.value
            assert chunk.source_name == doc.title
            assert len(chunk.section) > 0
            assert chunk.metadata["topic"] == "Generative AI"
            assert chunk.metadata["doc_id"] == "generative_ai"


# ── Vector Ingestion & Namespace Isolation Tests ───────────────────────


class TestKnowledgeBaseVectorIndexing:
    """Tests for vector ingestion into the knowledge_base partition."""

    def test_index_knowledge_base_directory(self, indexed_retrieval_service: RetrievalService):
        """Should index all chunks into the 'knowledge_base' partition."""
        kb_count = indexed_retrieval_service.vector_store.count("knowledge_base")

        assert kb_count >= 30  # 14 documents chunked into at least 30+ chunks
        assert "knowledge_base" in indexed_retrieval_service.vector_store.list_namespaces()

    def test_knowledge_base_is_isolated_from_resume_and_jd(
        self,
        indexed_retrieval_service: RetrievalService,
    ):
        """Adding resume chunks should not contaminate the knowledge_base partition."""
        resume_text = "John Doe — Python backend engineer with AWS experience."
        indexed_retrieval_service.index_resume(resume_text, filename="john_resume.pdf")

        assert indexed_retrieval_service.vector_store.count("resume") == 1
        assert indexed_retrieval_service.vector_store.count("knowledge_base") >= 30

        # Targeted search on knowledge base should not return resume chunk
        results = indexed_retrieval_service.retrieve_from_knowledge_base("Python backend", top_k=5)
        for r in results:
            assert r.source_type == CitationSource.KNOWLEDGE_BASE


# ── Semantic Retrieval Across 10 Core Domains ──────────────────────────


class TestDomainSemanticRetrieval:
    """Tests verifying accurate retrieval across each of the 10 required domains."""

    @pytest.mark.parametrize(
        "query,expected_topic_keyword",
        [
            ("What algorithms are used in supervised Machine Learning and tree ensembles?", "Machine Learning"),
            ("How does backpropagation and PyTorch autograd work in deep neural networks?", "Deep Learning"),
            ("What are Transformer self-attention mechanisms and tokenizers in NLP?", "Natural Language Processing"),
            ("How does YOLO object detection and OpenCV image filtering work?", "Computer Vision"),
            ("How do window functions, CTEs, and relational joins work in SQL?", "SQL"),
            ("How does asyncio event loop and FastAPI work in Python?", "Python"),
            ("How to handle data drift monitoring and model registries in MLOps?", "MLOps"),
            ("How does VPC networking, Terraform IaC, and AWS EC2 work in Cloud?", "Cloud"),
            ("How do Binary Search Trees, graphs, and dynamic programming work?", "Data Structures"),
            ("How does RAG architecture, LoRA fine-tuning, and prompt engineering work in GenAI?", "Generative AI"),
        ],
    )
    def test_retrieve_core_domains(
        self,
        indexed_retrieval_service: RetrievalService,
        query: str,
        expected_topic_keyword: str,
    ):
        """Querying for concepts in a domain should return highly relevant citations from that domain."""
        citations = indexed_retrieval_service.retrieve_from_knowledge_base(query, top_k=3)

        assert len(citations) > 0
        assert citations[0].relevance_score > 0.38
        assert any(
            expected_topic_keyword.lower() in c.source_name.lower() or
            expected_topic_keyword.lower() in c.content_preview.lower()
            for c in citations
        )

    def test_retrieve_learning_progression_section(self, indexed_retrieval_service: RetrievalService):
        """Searching for learning progression should retrieve the Suggested Learning Progression sections."""
        citations = indexed_retrieval_service.retrieve_from_knowledge_base(
            "What is the suggested learning progression and phases for MLOps?",
            top_k=4,
        )

        assert len(citations) > 0
        assert any("Learning Progression" in c.section or "Phase" in c.content_preview or "MLOps" in c.source_name for c in citations)

    def test_topic_filtered_retrieval(self, indexed_retrieval_service: RetrievalService):
        """retrieve_career_knowledge should filter by topic metadata when provided."""
        citations = indexed_retrieval_service.retrieve_career_knowledge(
            query="best practices and core tools",
            topic="MLOps",
            top_k=3,
        )

        assert len(citations) > 0
        assert all("MLOps" in c.source_name for c in citations)


# ── RAG Career Assistant Integration Tests ─────────────────────────────


class TestRAGCareerAssistantIntegration:
    """Tests integrating the curated knowledge base with RAGAssistant."""

    @pytest.fixture
    def assistant(self, indexed_retrieval_service: RetrievalService) -> RAGAssistant:
        """RAGAssistant with indexed knowledge base and mock LLM."""
        mock_llm = MockLLMProvider(
            default_response="Here is the recommended career roadmap based on our knowledge base.",
            model_name="mock-career-advisor",
        )
        return RAGAssistant(retrieval_service=indexed_retrieval_service, llm_provider=mock_llm)

    def test_recommend_career_path_generates_grounded_response(self, assistant: RAGAssistant):
        """recommend_career_path should retrieve knowledge base documents and ground the response."""
        response = assistant.recommend_career_path(
            career_goal_or_topic="Generative AI",
            candidate_skills=["Python", "PyTorch"],
            top_k=3,
        )

        assert response.has_sources is True
        assert len(response.sources) > 0
        assert all(c.source_type == CitationSource.KNOWLEDGE_BASE for c in response.sources)
        assert any("Generative AI" in c.source_name for c in response.sources)

    def test_recommend_learning_progression_generates_grounded_response(self, assistant: RAGAssistant):
        """recommend_learning_progression should retrieve phase-by-phase learning plans."""
        response = assistant.recommend_learning_progression("Computer Vision", top_k=3)

        assert response.has_sources is True
        assert len(response.sources) > 0
        assert any("Computer Vision" in c.source_name for c in response.sources)
