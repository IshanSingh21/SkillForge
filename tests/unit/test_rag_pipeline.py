"""
Tests for Milestone 7 — Complete RAG Pipeline & Career Assistant.

Tests cover:
- End-to-end RAG pipeline execution (Query → Embedding → FAISS → Context → Prompt → LLM → Citations)
- Anti-hallucination system prompt and context block formatting
- Domain-specific query helpers (ask_about_resume, ask_about_job, ask_career_advice)
- Graceful handling of empty retrieval results and empty queries
- Source citation provenance and relevance score propagation
- Pluggable LLM provider swapping (MockLLMProvider, factory resolution)
- Conversation history tracking and message trimming
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.ai.llm.base import LLMProvider, LLMResponse
from src.skillforge.ai.llm.factory import create_llm_provider
from src.skillforge.ai.llm.mock import MockLLMProvider
from src.skillforge.ai.vector_store import VectorStore
from src.skillforge.models.rag import CitationSource, ConversationRole, RAGResponse, SourceCitation
from src.skillforge.models.resume import TextChunk
from src.skillforge.services.rag_assistant import RAGAssistant
from src.skillforge.services.retrieval_service import RetrievalService


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine for RAG tests."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Configurable Mock LLM Provider."""
    return MockLLMProvider(
        default_response="Based on your resume, you have 5+ years of experience with Python and AWS.",
        model_name="test-mock-llm",
    )


@pytest.fixture
def retrieval_service(engine: EmbeddingEngine) -> RetrievalService:
    """Configured retrieval service populated with test data."""
    store = VectorStore(dimension=engine.dimension)
    service = RetrievalService(vector_store=store, embedding_engine=engine)

    # Ingest test resume
    resume_text = (
        "Jane Smith is a Senior Backend Engineer with 6 years of experience in Python, "
        "FastAPI, Docker, Kubernetes, PostgreSQL, and AWS cloud infrastructure."
    )
    service.index_resume(resume_text, filename="jane_smith_resume.pdf")

    # Ingest test job description
    jd_text = (
        "CloudTech Solutions is hiring a Senior Backend Engineer. Requirements include "
        "5+ years of backend development, Python, Go, microservices, and PostgreSQL database tuning."
    )
    service.index_job_description(jd_text, title="Senior Backend Engineer Job")

    # Ingest knowledge base
    kb_path = PROJECT_ROOT / "knowledge_base"
    if kb_path.exists():
        service.index_knowledge_base_directory(kb_path)

    return service


@pytest.fixture
def rag_assistant(retrieval_service: RetrievalService, mock_llm: MockLLMProvider) -> RAGAssistant:
    """Instantiated RAGAssistant with injected retrieval service and mock LLM."""
    return RAGAssistant(retrieval_service=retrieval_service, llm_provider=mock_llm)


# ── Full Pipeline Execution Tests ──────────────────────────────────────


class TestRAGPipelineExecution:
    """Tests for the complete end-to-end RAG query pipeline."""

    def test_end_to_end_query_returns_grounded_response(
        self,
        rag_assistant: RAGAssistant,
        mock_llm: MockLLMProvider,
    ):
        """Pipeline should retrieve context, format prompt, call LLM, and return citations."""
        response = rag_assistant.query("What cloud platforms does Jane have experience with?")

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert response.has_sources is True
        assert len(response.sources) > 0
        assert response.model_used == "test-mock-llm"
        assert 0.0 <= response.confidence <= 1.0

        # Verify LLM received context in prompt
        last_prompt = mock_llm.get_last_prompt()
        assert "--- RETRIEVED CONTEXT ---" in last_prompt
        assert "Jane Smith" in last_prompt
        assert "User Question:" in last_prompt

        # Verify anti-hallucination system prompt was passed
        last_system_prompt = mock_llm.get_last_system_prompt()
        assert "STRICTLY GROUNDED" in last_system_prompt
        assert "CRITICAL GROUNDING RULES" in last_system_prompt

    def test_sources_contain_valid_citations(self, rag_assistant: RAGAssistant):
        """Retrieved citations should carry valid metadata and relevance scores."""
        response = rag_assistant.query("Python and PostgreSQL experience")

        for citation in response.sources:
            assert isinstance(citation, SourceCitation)
            assert isinstance(citation.source_type, CitationSource)
            assert len(citation.source_name) > 0
            assert len(citation.content_preview) > 0
            assert citation.relevance_score > 0.0


# ── Domain-Specific Query Helpers ──────────────────────────────────────


class TestDomainSpecificQueries:
    """Tests for targeted retrieval queries (Resume only, JD only, KB only)."""

    def test_ask_about_resume_targets_resume_only(self, rag_assistant: RAGAssistant):
        """ask_about_resume should only retrieve citations from the 'resume' namespace."""
        response = rag_assistant.ask_about_resume("What frameworks does the candidate use?")

        assert response.has_sources is True
        assert all(c.source_type == CitationSource.RESUME for c in response.sources)

    def test_ask_about_job_targets_job_description_only(self, rag_assistant: RAGAssistant):
        """ask_about_job should only retrieve citations from the 'job_description' namespace."""
        response = rag_assistant.ask_about_job("What are the minimum required years of experience?")

        assert response.has_sources is True
        assert all(c.source_type == CitationSource.JOB_DESCRIPTION for c in response.sources)

    def test_ask_career_advice_targets_knowledge_base_only(self, rag_assistant: RAGAssistant):
        """ask_career_advice should only retrieve citations from the 'knowledge_base' namespace."""
        response = rag_assistant.ask_career_advice("How should I prepare using the STAR method?")

        assert response.has_sources is True
        assert all(c.source_type == CitationSource.KNOWLEDGE_BASE for c in response.sources)


# ── Edge Cases & Anti-Hallucination Guardrails ──────────────────────────


class TestRAGEdgeCases:
    """Tests for graceful error handling, empty queries, and out-of-domain questions."""

    def test_empty_query_handled_gracefully(self, rag_assistant: RAGAssistant):
        """Empty query should return guidance without calling the LLM."""
        response = rag_assistant.query("")

        assert "Please provide a question" in response.answer
        assert response.has_sources is False
        assert response.confidence == 0.0

    def test_out_of_domain_query_with_no_matches(self, rag_assistant: RAGAssistant, mock_llm: MockLLMProvider):
        """Query with completely unrelated topic should handle low/empty retrieval gracefully."""
        response = rag_assistant.query(
            "How do I bake a classic Italian pizza margherita with sourdough?",
            min_score=0.85,  # High threshold where no resume chunks match
        )

        assert "could not find any relevant information" in response.answer
        assert response.has_sources is False

    def test_conversation_history_updated_on_query(self, rag_assistant: RAGAssistant):
        """Each query should record both user and assistant messages in history."""
        initial_count = len(rag_assistant.history.messages)

        rag_assistant.query("What is Jane's primary programming language?")

        assert len(rag_assistant.history.messages) == initial_count + 2
        assert rag_assistant.history.messages[-2].role == ConversationRole.USER
        assert rag_assistant.history.messages[-1].role == ConversationRole.ASSISTANT


# ── Pluggable LLM Provider Tests ───────────────────────────────────────


class TestLLMProviderSwapping:
    """Tests verifying the LLM backend can be swapped cleanly."""

    def test_swap_llm_provider_at_runtime(self, retrieval_service: RetrievalService):
        """Should allow swapping between different LLM implementations without affecting retrieval."""
        custom_mock = MockLLMProvider(
            default_response="Custom swapped provider output.",
            model_name="swapped-custom-llm",
        )

        assistant = RAGAssistant(retrieval_service=retrieval_service, llm_provider=custom_mock)
        response = assistant.query("Python experience")

        assert response.answer == "Custom swapped provider output."
        assert response.model_used == "swapped-custom-llm"
        assert response.has_sources is True

    def test_create_mock_provider_via_factory(self):
        """Factory should instantiate MockLLMProvider with custom configuration."""
        provider = create_llm_provider("mock", default_response="Factory mock response", model_name="factory-mock")

        assert isinstance(provider, MockLLMProvider)
        assert provider.model_name == "factory-mock"

        res = provider.generate("Test prompt")
        assert res.content == "Factory mock response"
