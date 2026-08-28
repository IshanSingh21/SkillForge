"""
Tests for src.skillforge.services.resume_service — Resume processing pipeline.

Tests cover:
- Full pipeline (PDF bytes → ResumeAnalysis)
- Text-only pipeline (pasted text → ResumeAnalysis)
- Graceful failure handling
- Pipeline output structure
- Dependency injection
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.data.chunker import TextChunker
from src.skillforge.data.pdf_parser import PDFParser
from src.skillforge.data.preprocessor import TextPreprocessor
from src.skillforge.models.resume import ResumeAnalysis, TextChunk
from src.skillforge.services.resume_service import ResumeService
from src.skillforge.utils.exceptions import PDFParsingError, PreprocessingError


@pytest.fixture
def resume_service() -> ResumeService:
    """Return a ResumeService with default components."""
    return ResumeService(
        pdf_parser=PDFParser(),
        preprocessor=TextPreprocessor(),
        chunker=TextChunker(chunk_size=200, chunk_overlap=30),
    )


class TestResumeServicePDF:
    """Tests for ResumeService.process_resume() with PDF input."""

    def test_processes_valid_pdf(self, resume_service: ResumeService, minimal_pdf_bytes: bytes):
        """A valid PDF should produce a complete ResumeAnalysis."""
        result = resume_service.process_resume(minimal_pdf_bytes, filename="test.pdf")

        assert isinstance(result, ResumeAnalysis)
        assert result.filename == "test.pdf"
        assert result.page_count >= 1
        assert len(result.raw_text.strip()) > 0
        assert len(result.cleaned_text.strip()) > 0
        assert len(result.chunks) > 0

    def test_produces_chunks_from_pdf(self, resume_service: ResumeService, minimal_pdf_bytes: bytes):
        """Chunks should be TextChunk instances with source metadata."""
        result = resume_service.process_resume(minimal_pdf_bytes)

        for chunk in result.chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source == "resume"
            assert len(chunk.content) > 0

    def test_rejects_invalid_pdf(self, resume_service: ResumeService):
        """Non-PDF bytes should raise PDFParsingError."""
        with pytest.raises(PDFParsingError):
            resume_service.process_resume(b"not a pdf file", filename="bad.pdf")

    def test_rejects_empty_pdf(self, resume_service: ResumeService):
        """Empty bytes should raise PDFParsingError."""
        with pytest.raises(PDFParsingError):
            resume_service.process_resume(b"", filename="empty.pdf")

    def test_result_has_word_count(self, resume_service: ResumeService, minimal_pdf_bytes: bytes):
        """The word_count property should return a positive integer."""
        result = resume_service.process_resume(minimal_pdf_bytes)
        assert result.word_count > 0


class TestResumeServiceText:
    """Tests for ResumeService.process_text() with raw text input."""

    def test_processes_raw_text(self, resume_service: ResumeService, sample_resume_text: str):
        """Raw text should be processed into a ResumeAnalysis."""
        result = resume_service.process_text(sample_resume_text, source_name="pasted")

        assert isinstance(result, ResumeAnalysis)
        assert result.filename == "pasted"
        assert result.page_count == 0  # No PDF, so no pages
        assert len(result.cleaned_text) > 0
        assert len(result.chunks) > 0

    def test_detects_sections_from_text(self, resume_service: ResumeService, sample_resume_text: str):
        """Standard resume text should produce detected sections."""
        result = resume_service.process_text(sample_resume_text)

        assert len(result.sections) > 0
        section_titles = [s.title.lower() for s in result.sections]
        # Should detect at least one expected section
        assert any(
            keyword in title
            for title in section_titles
            for keyword in ["experience", "education", "skill"]
        )

    def test_produces_section_labeled_chunks(self, resume_service: ResumeService, sample_resume_text: str):
        """Chunks from sectioned text should carry section labels."""
        result = resume_service.process_text(sample_resume_text)

        sections_in_chunks = {c.section for c in result.chunks if c.section}
        assert len(sections_in_chunks) > 0  # At least some chunks are section-labeled

    def test_rejects_empty_text(self, resume_service: ResumeService):
        """Empty text should raise PreprocessingError."""
        with pytest.raises(PreprocessingError):
            resume_service.process_text("")

    def test_rejects_whitespace_only_text(self, resume_service: ResumeService):
        """Whitespace-only text should raise PreprocessingError."""
        with pytest.raises(PreprocessingError):
            resume_service.process_text("   \n\n   ")

    def test_handles_text_without_sections(self, resume_service: ResumeService):
        """Plain text without section headers should still produce chunks."""
        result = resume_service.process_text(
            "I am an experienced Python developer with 5 years "
            "building web applications using Django and FastAPI."
        )

        assert len(result.sections) == 0
        assert len(result.chunks) > 0  # Full-text chunking should work

    def test_cleaned_text_preserves_content(self, resume_service: ResumeService, sample_resume_text: str):
        """Key information should survive the cleaning process."""
        result = resume_service.process_text(sample_resume_text)

        # Core content should still be present
        assert "Python" in result.cleaned_text
        assert "Acme Corp" in result.cleaned_text or "Software Engineer" in result.cleaned_text


class TestResumeServiceDI:
    """Tests for dependency injection and configuration."""

    def test_uses_injected_chunker(self, sample_resume_text: str):
        """A custom chunker should affect chunk output."""
        tiny_chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        service = ResumeService(chunker=tiny_chunker)

        result = service.process_text(sample_resume_text)

        # With an 80-char chunk size, we should get many more chunks
        # than with the default 512
        assert len(result.chunks) > 10

    def test_default_construction(self):
        """ResumeService should work with all defaults."""
        service = ResumeService()
        assert service.pdf_parser is not None
        assert service.preprocessor is not None
        assert service.chunker is not None


class TestProcessingErrors:
    """Tests for the fail-soft warning collection."""

    def test_warnings_collected_in_result(self, resume_service: ResumeService, minimal_pdf_bytes: bytes):
        """Processing errors list should be present (even if empty)."""
        result = resume_service.process_resume(minimal_pdf_bytes)
        assert isinstance(result.processing_errors, list)

    def test_text_pipeline_collects_errors_gracefully(self, resume_service: ResumeService):
        """Non-fatal issues should appear as warnings, not exceptions."""
        # Short text that works but might generate warnings
        result = resume_service.process_text("Valid but very short resume text here.")
        assert isinstance(result.processing_errors, list)
        # Should still produce a result even if sections weren't found
        assert result.cleaned_text is not None
