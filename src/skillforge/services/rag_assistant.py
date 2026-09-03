"""
SkillForge AI — Complete RAG Career Assistant Pipeline.

Executes the end-to-end Retrieval-Augmented Generation (RAG) pipeline:
    User Query
    → Query Embedding (SentenceTransformers all-MiniLM-L6-v2)
    → FAISS Vector Retrieval (partitioned across Resume, JD, Knowledge Base)
    → Context Selection & Provenance Formatting
    → Grounding Prompt Construction with Anti-Hallucination Directives
    → Pluggable LLM Backend (Gemini / Groq / Mock)
    → Grounded Response with Source Citations

Design Principles:
    1. Strict Grounding: System prompts explicitly forbid inventing facts. If context is
       insufficient, the assistant explicitly reports that information is missing.
    2. Decoupled Architecture: Retrieval and generation are separate modules.
       The LLM provider can be swapped at runtime without altering retrieval logic.
    3. Source Provenance: Every response is accompanied by structured SourceCitation
       objects linking facts directly to the specific source document and section.
    4. Graceful Fallbacks: Handles empty vector stores, empty query inputs, and provider
       failures with meaningful feedback.

Usage:
    from src.skillforge.services.rag_assistant import RAGAssistant

    assistant = RAGAssistant()
    # Ingest context into vector store
    assistant.retrieval_service.index_resume(resume_text)
    assistant.retrieval_service.index_job_description(jd_text)

    response = assistant.query("What experience does the candidate have with Kubernetes?")
    print(response.answer)
    for src in response.sources:
        print(f" - [{src.source_name}] ({src.relevance_score:.2f}): {src.content_preview}")
"""

from __future__ import annotations

from typing import Any

from src.skillforge.ai.llm.base import LLMProvider
from src.skillforge.ai.llm.factory import create_llm_provider
from src.skillforge.models.rag import (
    CitationSource,
    ConversationHistory,
    ConversationRole,
    RAGResponse,
    SourceCitation,
)
from src.skillforge.services.retrieval_service import RetrievalService
from src.skillforge.utils.exceptions import LLMError, RAGError
from src.skillforge.utils.logging import logger


class RAGAssistant:
    """
    Retrieval-Augmented Generation assistant for career and resume intelligence.

    Combines vector similarity search across resumes, job descriptions, and
    curated career guides with LLM text generation.
    """

    # System prompt enforcing strict factuality and anti-hallucination guardrails
    SYSTEM_PROMPT = """You are SkillForge AI, an expert, objective career and technical resume assistant.

Your primary directive is to provide helpful, professional, and accurate answers STRICTLY GROUNDED in the provided context documents (Candidate Resume, Job Description, and Career Knowledge Base).

CRITICAL GROUNDING RULES:
1. Base your answer ONLY on the facts directly mentioned in the "RETRIEVED CONTEXT" section below.
2. If the retrieved context does NOT contain enough information to answer the question, explicitly state:
   "Based on the provided documents, I do not have enough information to answer this question."
   Do NOT attempt to guess, assume, or invent credentials, experience, or job requirements.
3. When referencing specific details (such as tools, years of experience, or responsibilities), refer to the source document (e.g. "[Resume]", "[Job Description]", or "[Interview Guide]").
4. Be concise, structured, and professional in your response. Use markdown bullet points for readability.
"""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_provider: LLMProvider | None = None,
        history: ConversationHistory | None = None,
    ) -> None:
        """
        Initialize the RAG Assistant with dependency injection.

        Args:
            retrieval_service: Multi-source vector retrieval service.
            llm_provider: LLM backend provider (Gemini, Groq, Mock).
            history: Optional conversation history tracker.
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider or create_llm_provider()
        self.history = history or ConversationHistory()

    # ── Main Query Pipeline ────────────────────────────────────────────

    def query(
        self,
        question: str,
        sources: list[CitationSource | str] | None = None,
        top_k: int = 4,
        min_score: float = 0.20,
        session_context: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Execute the complete end-to-end RAG pipeline.

        Pipeline Stages:
            1. Query Validation
            2. Vector Retrieval across specified sources
            3. Context Formatting & Prompt Construction
            4. LLM Generation
            5. Response Assembly with Source Citations

        Args:
            question: User question or career inquiry.
            sources: Optional list of sources to search ('resume', 'job_description', 'knowledge_base').
            top_k: Number of context chunks to retrieve.
            min_score: Minimum similarity score threshold.
            session_context: Optional additional metadata (e.g. candidate name, target role).

        Returns:
            RAGResponse with grounded text and citations.
        """
        clean_question = question.strip() if question else ""
        if not clean_question:
            return RAGResponse(
                answer="Please provide a question to query your resume or career knowledge base.",
                sources=[],
                query=question,
                model_used=getattr(self.llm_provider, "model_name", "none"),
                confidence=0.0,
            )

        logger.info("Executing RAG query", query=clean_question[:80])

        try:
            # Stage 1: Retrieve relevant source citations
            citations = self.retrieval_service.retrieve(
                query=clean_question,
                sources=sources,
                top_k=top_k,
                min_score=min_score,
            )

            # Stage 2: Handle empty retrieval gracefully
            if not citations:
                logger.info("No relevant context found for query", query=clean_question)
                return RAGResponse(
                    answer=(
                        "Based on the provided documents, I could not find any relevant information "
                        "matching your query. Please ensure your resume or job description has been uploaded, "
                        "or try rephrasing your question."
                    ),
                    sources=[],
                    query=clean_question,
                    model_used=getattr(self.llm_provider, "model_name", "none"),
                    confidence=0.0,
                )

            # Stage 3: Construct Grounded Prompt
            context_block = self._format_retrieved_context(citations)
            user_prompt = self._build_user_prompt(
                question=clean_question,
                context_block=context_block,
                session_context=session_context,
            )

            # Stage 4: Call LLM Provider
            llm_res = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.3,  # Low temperature to prioritize factuality
            )

            # Stage 5: Calculate response confidence
            avg_score = sum(c.relevance_score for c in citations) / len(citations)
            confidence = float(min(1.0, max(0.0, avg_score)))

            # Stage 6: Update Conversation History
            self.history.add_message(
                role=ConversationRole.USER,
                content=clean_question,
            )
            self.history.add_message(
                role=ConversationRole.ASSISTANT,
                content=llm_res.content,
                sources=citations,
            )

            return RAGResponse(
                answer=llm_res.content.strip(),
                sources=citations,
                query=clean_question,
                model_used=llm_res.model or getattr(self.llm_provider, "model_name", "llm"),
                confidence=round(confidence, 3),
            )

        except LLMError as e:
            logger.error("LLM generation error in RAG pipeline", error=str(e))
            raise RAGError(f"RAG Assistant encountered an LLM error: {e}") from e
        except Exception as e:
            logger.error("Unexpected error in RAG pipeline", error=str(e))
            raise RAGError(f"RAG pipeline failed: {e}") from e

    # ── Specialized Domain Queries ─────────────────────────────────────

    def ask_about_resume(self, question: str, top_k: int = 3) -> RAGResponse:
        """Query specifically against the candidate's uploaded resume."""
        return self.query(
            question=question,
            sources=[CitationSource.RESUME],
            top_k=top_k,
        )

    def ask_about_job(self, question: str, top_k: int = 3) -> RAGResponse:
        """Query specifically against the target job description."""
        return self.query(
            question=question,
            sources=[CitationSource.JOB_DESCRIPTION],
            top_k=top_k,
        )

    def ask_career_advice(self, question: str, top_k: int = 3) -> RAGResponse:
        """Query specifically against the career guides knowledge base."""
        return self.query(
            question=question,
            sources=[CitationSource.KNOWLEDGE_BASE],
            top_k=top_k,
        )

    def recommend_career_path(
        self,
        career_goal_or_topic: str,
        candidate_skills: list[str] | None = None,
        top_k: int = 4,
    ) -> RAGResponse:
        """
        Generate grounded career path recommendations for a target role or topic.

        Args:
            career_goal_or_topic: Target domain or goal (e.g. 'MLOps', 'Computer Vision').
            candidate_skills: Optional list of candidate's current skills.
            top_k: Number of knowledge chunks to retrieve.

        Returns:
            RAGResponse grounded in career knowledge base documents.
        """
        prompt = f"What are the key career roles, prerequisite skills, and typical project experience for {career_goal_or_topic}?"
        if candidate_skills:
            prompt += f" The candidate currently has skills in: {', '.join(candidate_skills)}."

        return self.query(
            question=prompt,
            sources=[CitationSource.KNOWLEDGE_BASE],
            top_k=top_k,
        )

    def recommend_learning_progression(
        self,
        target_skill_or_role: str,
        top_k: int = 4,
    ) -> RAGResponse:
        """
        Retrieve structured learning phases and milestone progression for a technical area.

        Args:
            target_skill_or_role: Topic (e.g. 'Generative AI', 'Deep Learning', 'SQL').
            top_k: Number of knowledge chunks to retrieve.

        Returns:
            RAGResponse containing structured phase-by-phase learning plan.
        """
        prompt = f"What is the recommended phase-by-phase learning progression and core concepts to master for {target_skill_or_role}?"
        return self.query(
            question=prompt,
            sources=[CitationSource.KNOWLEDGE_BASE],
            top_k=top_k,
        )

    # ── Context & Prompt Formatting Helpers ────────────────────────────

    def _format_retrieved_context(self, citations: list[SourceCitation]) -> str:
        """
        Format retrieved chunks into a clearly demarcated context block.

        Includes document name, section title, and relevance score for each chunk.
        """
        blocks = []
        for i, c in enumerate(citations, 1):
            source_label = c.source_type.value.replace("_", " ").title()
            section_info = f" | Section: {c.section}" if c.section else ""
            header = f"[Document {i}: {source_label} - '{c.source_name}'{section_info} | Relevance: {c.relevance_score:.2f}]"
            blocks.append(f"{header}\n{c.content_preview}")

        return "\n\n".join(blocks)

    def _build_user_prompt(
        self,
        question: str,
        context_block: str,
        session_context: dict[str, Any] | None = None,
    ) -> str:
        """Construct the prompt sent to the LLM containing context and question."""
        prompt_parts = []

        if session_context:
            role = session_context.get("target_role")
            if role:
                prompt_parts.append(f"Target Role Context: {role}\n")

        prompt_parts.append("--- RETRIEVED CONTEXT ---")
        prompt_parts.append(context_block)
        prompt_parts.append("--- END OF CONTEXT ---\n")
        prompt_parts.append(f"User Question: {question}")
        prompt_parts.append("\nAnswer based ONLY on the context provided above:")

        return "\n".join(prompt_parts)
