"""
SkillForge AI — Resume Processing Service.

Orchestrates the full resume processing pipeline: PDF extraction →
text cleaning → section detection → skill extraction → text chunking.
This is the single entry point that the UI calls to process an uploaded resume.

Design decisions:
    - Dependencies (parser, preprocessor, chunker, skill extractor) are injected
      via the constructor so each component can be tested independently or swapped.
    - The pipeline is deliberately fail-soft: individual stage failures
      are captured as warnings in the result rather than aborting the
      entire pipeline, so users get partial results when possible.
    - Chunking is done both per-section (for targeted retrieval) and on
      the full text (as a fallback when section detection fails).

Usage:
    from src.skillforge.services.resume_service import ResumeService

    service = ResumeService()
    analysis = service.process_resume(pdf_bytes, filename="resume.pdf")
"""

from __future__ import annotations

from config.settings import get_settings
from src.skillforge.data.chunker import TextChunker
from src.skillforge.data.pdf_parser import PDFParser
from src.skillforge.data.preprocessor import TextPreprocessor
from src.skillforge.models.resume import ResumeAnalysis, Skill, TextChunk
from src.skillforge.services.skill_extractor import SkillExtractor
from src.skillforge.utils.exceptions import (
    ChunkingError,
    PDFParsingError,
    PreprocessingError,
    SkillExtractionError,
    SkillForgeError,
)
from src.skillforge.utils.logging import logger


class ResumeService:
    """
    Orchestrates the resume processing pipeline.

    Pipeline stages:
        1. PDF text extraction (PyMuPDF)
        2. Text cleaning & normalization
        3. Section detection (Experience, Education, Skills, …)
        4. Skill extraction (Taxonomy + NLP)
        5. Text chunking for embedding/retrieval

    Each stage feeds into the next. Non-fatal failures at any stage
    produce warnings but do not abort the pipeline.
    """

    def __init__(
        self,
        pdf_parser: PDFParser | None = None,
        preprocessor: TextPreprocessor | None = None,
        chunker: TextChunker | None = None,
        skill_extractor: SkillExtractor | None = None,
    ) -> None:
        """
        Initialize with optional dependency injection.

        If components are not provided, defaults are created using
        application settings.
        """
        settings = get_settings()

        self.pdf_parser = pdf_parser or PDFParser()
        self.preprocessor = preprocessor or TextPreprocessor()
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.skill_extractor = skill_extractor or SkillExtractor(use_nlp=True)

    def process_resume(
        self,
        pdf_bytes: bytes,
        filename: str = "",
    ) -> ResumeAnalysis:
        """
        Process a resume PDF through the full pipeline.

        Args:
            pdf_bytes: Raw bytes of the uploaded PDF file.
            filename: Original filename (for metadata).

        Returns:
            ResumeAnalysis containing raw text, cleaned text, sections,
            extracted skills, and text chunks ready for embedding.

        Raises:
            PDFParsingError: If PDF extraction completely fails.
            PreprocessingError: If text cleaning completely fails.
        """
        logger.info("Starting resume processing", filename=filename)
        processing_errors: list[str] = []

        # ── Stage 1: PDF Extraction ────────────────────────────────
        extraction = self.pdf_parser.extract_text(pdf_bytes)
        raw_text = extraction.text
        page_count = extraction.page_count
        processing_errors.extend(extraction.warnings)

        logger.info(
            "PDF extraction complete",
            pages=page_count,
            chars=len(raw_text),
            warnings=len(extraction.warnings),
        )

        # ── Stage 2: Text Cleaning ─────────────────────────────────
        try:
            cleaned_text = self.preprocessor.clean_text(raw_text)
        except PreprocessingError:
            # If cleaning fails, fall back to lightly stripped raw text
            logger.warning("Text cleaning failed, using raw text")
            cleaned_text = raw_text.strip()
            processing_errors.append("Text cleaning failed — using raw extracted text.")

        logger.info(
            "Text cleaning complete",
            raw_chars=len(raw_text),
            cleaned_chars=len(cleaned_text),
        )

        # ── Stage 3: Section Detection ─────────────────────────────
        try:
            sections = self.preprocessor.extract_sections(cleaned_text)
        except Exception as e:
            logger.warning("Section extraction failed", error=str(e))
            sections = []
            processing_errors.append(f"Section detection failed: {e}")

        logger.info("Section detection complete", sections_found=len(sections))

        # ── Stage 4: Skill Extraction ──────────────────────────────
        skills: list[Skill] = []
        try:
            skills = self.skill_extractor.extract_skills(cleaned_text, source="resume")
            logger.info("Skill extraction complete", num_skills=len(skills))
        except Exception as e:
            logger.warning("Skill extraction in resume pipeline failed", error=str(e))
            processing_errors.append(f"Skill extraction warning: {e}")

        # ── Stage 5: Text Chunking ─────────────────────────────────
        chunks = self._generate_chunks(
            cleaned_text=cleaned_text,
            sections=sections,
            filename=filename,
            processing_errors=processing_errors,
        )

        logger.info("Chunking complete", num_chunks=len(chunks))

        # ── Build Result ───────────────────────────────────────────
        analysis = ResumeAnalysis(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            sections=sections,
            skills=skills,
            chunks=chunks,
            filename=filename,
            page_count=page_count,
            processing_errors=processing_errors,
        )

        logger.info(
            "Resume processing complete",
            filename=filename,
            pages=page_count,
            sections=len(sections),
            skills=len(skills),
            chunks=len(chunks),
            words=analysis.word_count,
            warnings=len(processing_errors),
        )

        return analysis

    def process_text(
        self,
        raw_text: str,
        source_name: str = "pasted_text",
    ) -> ResumeAnalysis:
        """
        Process raw text (pasted by user) through the cleaning + chunking
        pipeline, bypassing PDF extraction.

        This is the fallback for users who can't upload a PDF.

        Args:
            raw_text: Raw resume text pasted by the user.
            source_name: Label for the text source.

        Returns:
            ResumeAnalysis with cleaned text, sections, skills, and chunks.

        Raises:
            PreprocessingError: If text is empty or cleaning fails.
        """
        logger.info("Processing pasted resume text", chars=len(raw_text))
        processing_errors: list[str] = []

        cleaned_text = self.preprocessor.clean_text(raw_text)

        try:
            sections = self.preprocessor.extract_sections(cleaned_text)
        except Exception as e:
            sections = []
            processing_errors.append(f"Section detection failed: {e}")

        # Extract skills
        skills: list[Skill] = []
        try:
            skills = self.skill_extractor.extract_skills(cleaned_text, source="resume")
        except Exception as e:
            logger.warning("Skill extraction from text failed", error=str(e))
            processing_errors.append(f"Skill extraction warning: {e}")

        chunks = self._generate_chunks(
            cleaned_text=cleaned_text,
            sections=sections,
            filename=source_name,
            processing_errors=processing_errors,
        )

        return ResumeAnalysis(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            sections=sections,
            skills=skills,
            chunks=chunks,
            filename=source_name,
            page_count=0,
            processing_errors=processing_errors,
        )

    def _generate_chunks(
        self,
        cleaned_text: str,
        sections: list,
        filename: str,
        processing_errors: list[str],
    ) -> list[TextChunk]:
        """
        Generate text chunks from the cleaned text.

        Strategy:
            - If sections were detected, chunk each section independently
              so that each chunk carries its section label (useful for
              targeted retrieval, e.g. "search only in Experience").
            - Always also chunk the full text as a fallback collection.
            - Deduplicate by content to avoid redundant embeddings.
        """
        chunks: list[TextChunk] = []

        # Chunk per-section if sections were detected
        if sections:
            try:
                section_dicts = [
                    {"title": s.title, "content": s.content} for s in sections
                ]
                section_chunks = self.chunker.chunk_sections(
                    sections=section_dicts,
                    source="resume",
                    source_name=filename,
                )
                chunks.extend(section_chunks)
            except ChunkingError as e:
                logger.warning("Section-level chunking failed", error=str(e))
                processing_errors.append(f"Section chunking failed: {e}")

        # Always chunk full text as fallback / supplement
        try:
            full_chunks = self.chunker.chunk_text(
                text=cleaned_text,
                source="resume",
                source_name=filename,
                section="full_document",
            )
            # Deduplicate: only add full-text chunks whose content
            # isn't already covered by a section chunk
            existing_content = {c.content for c in chunks}
            for chunk in full_chunks:
                if chunk.content not in existing_content:
                    chunks.append(chunk)
        except ChunkingError as e:
            logger.warning("Full-text chunking failed", error=str(e))
            processing_errors.append(f"Full-text chunking failed: {e}")

        return chunks
