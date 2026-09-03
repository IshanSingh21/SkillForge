"""
SkillForge AI — Vector Retrieval Service.

Orchestrates document chunking, dense embedding generation, FAISS indexing,
and multi-source semantic retrieval for the SkillForge RAG infrastructure.

Supports 3 primary knowledge partitions:
    1. Resume ('resume') — candidate's history, projects, and skills
    2. Job Description ('job_description') — role requirements and qualifications
    3. Career Knowledge Base ('knowledge_base') — curated career guides, interview strategies,
       domain-specific roadmaps (ML, DL, NLP, CV, SQL, Python, MLOps, Cloud, DSA, GenAI)

Design Decisions:
    - Chunking strategy: 512-character chunks with 50-character overlap
      balanced for `all-MiniLM-L6-v2` (256-token limit), preserving complete
      bullet points/paragraphs without truncation.
    - Uniform output: All retrieval queries return strongly typed `SourceCitation` models
      with document provenance, section name, relevance score, and content snippets.
    - Decoupled architecture: Independent of Streamlit UI and LLM generator,
      usable by automated tests, scripts, and future conversational endpoints.

Usage:
    from src.skillforge.services.retrieval_service import RetrievalService

    service = RetrievalService()
    service.index_resume(resume_text, filename="resume.pdf")
    service.index_job_description(jd_text, title="Senior Backend Engineer")
    service.index_knowledge_base_directory()

    citations = service.retrieve("What system design experience is required?", top_k=3)
    for c in citations:
        print(f"[{c.source_type.value}] {c.content_preview} (Score: {c.relevance_score:.2f})")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.ai.vector_store import SearchResult, VectorStore
from src.skillforge.data.chunker import TextChunker
from src.skillforge.data.knowledge_base_loader import KnowledgeBaseLoader
from src.skillforge.models.rag import CitationSource, SourceCitation
from src.skillforge.models.resume import ResumeAnalysis, TextChunk
from src.skillforge.utils.exceptions import ChunkingError, EmbeddingError, VectorStoreError
from src.skillforge.utils.logging import logger


class RetrievalService:
    """
    Multi-source vector indexing and similarity retrieval service.

    Connects TextChunker, EmbeddingEngine, KnowledgeBaseLoader, and FAISS VectorStore
    into a unified retrieval system.
    """

    # Standard default chunking parameters
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        chunker: TextChunker | None = None,
        kb_loader: KnowledgeBaseLoader | None = None,
    ) -> None:
        """
        Initialize the retrieval service with dependency injection.

        Args:
            vector_store: FAISS VectorStore instance (or created with default dimension 384).
            embedding_engine: SentenceTransformer engine.
            chunker: Document chunker configured with standard chunk size and overlap.
            kb_loader: KnowledgeBaseLoader instance for parsing markdown documents.
        """
        self.engine = embedding_engine or EmbeddingEngine()
        self.chunker = chunker or TextChunker(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP,
        )
        self.kb_loader = kb_loader or KnowledgeBaseLoader()
        self.vector_store = vector_store or VectorStore(dimension=self.engine.dimension)

    # ── Indexing Methods ───────────────────────────────────────────────

    def index_resume(
        self,
        resume: ResumeAnalysis | str,
        filename: str = "resume.pdf",
    ) -> int:
        """
        Chunk, embed, and index a candidate's resume in the 'resume' namespace.

        Args:
            resume: ResumeAnalysis object or raw resume text.
            filename: Name/title of the resume document.

        Returns:
            Number of chunks indexed.
        """
        chunks: list[TextChunk] = []

        if isinstance(resume, ResumeAnalysis):
            if resume.chunks:
                chunks = resume.chunks
            else:
                text = resume.cleaned_text or resume.raw_text
                chunks = self.chunker.chunk_text(
                    text=text,
                    source=CitationSource.RESUME.value,
                    source_name=resume.filename or filename,
                    section="full_document",
                )
        else:
            chunks = self.chunker.chunk_text(
                text=str(resume),
                source=CitationSource.RESUME.value,
                source_name=filename,
                section="full_document",
            )

        if not chunks:
            logger.warning("No chunks generated for resume indexing")
            return 0

        # Generate embeddings for chunks
        texts_to_embed = [c.content for c in chunks]
        embeddings = self.engine.encode(texts_to_embed)

        self.vector_store.upsert(
            namespace=CitationSource.RESUME.value,
            chunks=chunks,
            embeddings=embeddings,
        )

        logger.info(
            "Resume indexed in vector store",
            chunk_count=len(chunks),
            filename=filename,
        )
        return len(chunks)

    def index_job_description(
        self,
        job_description: str,
        title: str = "Target Job Description",
    ) -> int:
        """
        Chunk, embed, and index a job description in the 'job_description' namespace.

        Args:
            job_description: Full job description text.
            title: Title or role identifier.

        Returns:
            Number of chunks indexed.
        """
        if not job_description or not job_description.strip():
            logger.warning("Empty job description provided for indexing")
            return 0

        chunks = self.chunker.chunk_text(
            text=job_description,
            source=CitationSource.JOB_DESCRIPTION.value,
            source_name=title,
            section="requirements",
        )

        if not chunks:
            return 0

        texts_to_embed = [c.content for c in chunks]
        embeddings = self.engine.encode(texts_to_embed)

        self.vector_store.upsert(
            namespace=CitationSource.JOB_DESCRIPTION.value,
            chunks=chunks,
            embeddings=embeddings,
        )

        logger.info(
            "Job description indexed in vector store",
            chunk_count=len(chunks),
            title=title,
        )
        return len(chunks)

    def index_knowledge_base_directory(
        self,
        directory: str | Path | None = None,
    ) -> int:
        """
        Read, parse with KnowledgeBaseLoader, chunk, embed, and index all markdown files
        from the knowledge base directory into the 'knowledge_base' namespace.

        Args:
            directory: Path to knowledge_base folder. If None, auto-detects from project root.

        Returns:
            Total number of knowledge base chunks indexed.
        """
        if directory is None:
            # Default to project root / knowledge_base
            kb_dir = Path(__file__).resolve().parent.parent.parent.parent / "knowledge_base"
        else:
            kb_dir = Path(directory)

        if not kb_dir.exists() or not kb_dir.is_dir():
            logger.warning("Knowledge base directory not found", path=str(kb_dir))
            return 0

        # Load and section-chunk all markdown files using KnowledgeBaseLoader
        all_chunks = self.kb_loader.chunk_directory(
            directory_path=kb_dir,
            chunker=self.chunker,
            recursive=True,
        )

        if not all_chunks:
            logger.warning("No knowledge base chunks found to index")
            return 0

        texts_to_embed = [c.content for c in all_chunks]
        embeddings = self.engine.encode(texts_to_embed)

        self.vector_store.upsert(
            namespace=CitationSource.KNOWLEDGE_BASE.value,
            chunks=all_chunks,
            embeddings=embeddings,
        )

        logger.info(
            "Knowledge base indexed in vector store",
            total_chunks=len(all_chunks),
            directory=str(kb_dir),
        )
        return len(all_chunks)

    # ── Retrieval Methods ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        sources: list[str | CitationSource] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceCitation]:
        """
        Retrieve relevant source citations across specified or all namespaces.

        Args:
            query: Natural language question or search query.
            sources: List of sources to search ('resume', 'job_description', 'knowledge_base').
                     If None, searches across all active indexed sources.
            top_k: Maximum number of citations to return.
            min_score: Minimum similarity score threshold (0.0 to 1.0).
            filter_metadata: Optional metadata filter.

        Returns:
            List of SourceCitation models sorted by relevance descending.
        """
        if not query or not query.strip():
            return []

        query_emb = self.engine.encode_single(query)
        all_results: list[SearchResult] = []

        if sources:
            # Search each requested namespace
            for s in sources:
                ns_name = s.value if isinstance(s, CitationSource) else str(s).lower()
                res = self.vector_store.search(
                    query_embedding=query_emb,
                    namespace=ns_name,
                    top_k=top_k,
                    min_score=min_score,
                    filter_metadata=filter_metadata,
                )
                all_results.extend(res)

            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[:top_k]
        else:
            # Search all namespaces
            all_results = self.vector_store.search(
                query_embedding=query_emb,
                namespace="",  # empty string = all namespaces
                top_k=top_k,
                min_score=min_score,
                filter_metadata=filter_metadata,
            )

        # Convert to SourceCitation models
        citations: list[SourceCitation] = []
        for res in all_results:
            chunk = res.chunk
            source_type_enum = self._map_source_type(chunk.source)
            preview = chunk.content.strip()
            if len(preview) > 400:
                preview = preview[:397] + "..."

            citation = SourceCitation(
                source_type=source_type_enum,
                source_name=chunk.source_name,
                section=chunk.section,
                content_preview=preview,
                relevance_score=round(res.score, 3),
                chunk_id=chunk.chunk_id,
            )
            citations.append(citation)

        return citations

    def retrieve_from_resume(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SourceCitation]:
        """Convenience method to retrieve citations strictly from the candidate's resume."""
        return self.retrieve(
            query=query,
            sources=[CitationSource.RESUME],
            top_k=top_k,
            min_score=min_score,
        )

    def retrieve_from_job_description(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SourceCitation]:
        """Convenience method to retrieve citations strictly from the target job description."""
        return self.retrieve(
            query=query,
            sources=[CitationSource.JOB_DESCRIPTION],
            top_k=top_k,
            min_score=min_score,
        )

    def retrieve_from_knowledge_base(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SourceCitation]:
        """Convenience method to retrieve citations strictly from the career knowledge base."""
        return self.retrieve(
            query=query,
            sources=[CitationSource.KNOWLEDGE_BASE],
            top_k=top_k,
            min_score=min_score,
        )

    def retrieve_career_knowledge(
        self,
        query: str,
        topic: str | None = None,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> list[SourceCitation]:
        """
        Retrieve career knowledge with optional topic-level filtering.

        Args:
            query: Career inquiry or technology question.
            topic: Optional topic filter (e.g. 'Machine Learning', 'Python', 'MLOps').
            top_k: Number of citations.
            min_score: Score threshold.

        Returns:
            List of SourceCitation models.
        """
        filter_meta = {"topic": topic} if topic else None
        return self.retrieve(
            query=query,
            sources=[CitationSource.KNOWLEDGE_BASE],
            top_k=top_k,
            min_score=min_score,
            filter_metadata=filter_meta,
        )

    # ── Status & State ─────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return the current indexing status across all namespaces."""
        namespaces = self.vector_store.list_namespaces()
        counts = {ns: self.vector_store.count(ns) for ns in namespaces}
        return {
            "total_chunks": self.vector_store.count(),
            "namespaces": counts,
            "dimension": self.vector_store.dimension,
        }

    def clear(self, source: str | CitationSource | None = None) -> None:
        """Clear all indexed data or a specific source partition."""
        if source:
            ns_name = source.value if isinstance(source, CitationSource) else str(source)
            self.vector_store.clear(namespace=ns_name)
        else:
            self.vector_store.clear()

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _map_source_type(source_str: str) -> CitationSource:
        """Map source string to CitationSource enum with fallback."""
        s = source_str.lower().strip()
        if "resume" in s:
            return CitationSource.RESUME
        if "job" in s or "jd" in s:
            return CitationSource.JOB_DESCRIPTION
        return CitationSource.KNOWLEDGE_BASE
