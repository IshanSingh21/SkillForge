"""
Integration test: Full resume processing pipeline.

Tests the complete flow from PDF bytes through extraction,
cleaning, section detection, and chunking — verifying that
all components work correctly together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.data.chunker import TextChunker
from src.skillforge.data.pdf_parser import PDFParser
from src.skillforge.data.preprocessor import TextPreprocessor
from src.skillforge.models.resume import ResumeAnalysis
from src.skillforge.services.resume_service import ResumeService


@pytest.fixture
def pipeline_service() -> ResumeService:
    """Build a full pipeline with explicit small chunks for testing."""
    return ResumeService(
        pdf_parser=PDFParser(),
        preprocessor=TextPreprocessor(),
        chunker=TextChunker(chunk_size=150, chunk_overlap=20),
    )


class TestFullPipeline:
    """End-to-end integration tests for the resume processing pipeline."""

    def test_pdf_to_analysis_roundtrip(self, pipeline_service: ResumeService, minimal_pdf_bytes: bytes):
        """
        A PDF should flow through the entire pipeline and produce
        a fully populated ResumeAnalysis.
        """
        result = pipeline_service.process_resume(minimal_pdf_bytes, filename="integration_test.pdf")

        # Verify the result is fully populated
        assert isinstance(result, ResumeAnalysis)
        assert result.filename == "integration_test.pdf"
        assert result.page_count >= 1
        assert len(result.raw_text) > 0
        assert len(result.cleaned_text) > 0
        assert len(result.chunks) > 0

        # Verify chunk integrity
        for chunk in result.chunks:
            assert chunk.source == "resume"
            assert chunk.source_name == "integration_test.pdf"
            assert chunk.chunk_id is not None
            assert len(chunk.content) > 0

    def test_text_pipeline_with_rich_resume(self, pipeline_service: ResumeService, sample_resume_text: str):
        """
        A rich resume text with multiple sections should produce
        section-labeled chunks and detected sections.
        """
        result = pipeline_service.process_text(sample_resume_text, source_name="full_resume")

        # Should detect multiple sections
        assert len(result.sections) >= 3  # At minimum: Summary/Experience/Education/Skills

        # Should produce multiple chunks due to small chunk_size
        assert len(result.chunks) >= 3

        # At least some chunks should have section labels
        labeled_chunks = [c for c in result.chunks if c.section and c.section != "full_document"]
        assert len(labeled_chunks) > 0

        # Original content should survive the pipeline
        assert "Python" in result.cleaned_text
        assert result.word_count > 50

    def test_pipeline_components_are_decoupled(self):
        """
        Each pipeline component should work independently — replacing
        one shouldn't break the others.
        """
        # Build pipeline with custom chunker
        custom_chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        service = ResumeService(
            pdf_parser=PDFParser(),
            preprocessor=TextPreprocessor(),
            chunker=custom_chunker,
        )

        result = service.process_text("A simple test of component independence.")
        assert len(result.chunks) > 0

    def test_processing_is_idempotent(self, pipeline_service: ResumeService, sample_resume_text: str):
        """Processing the same text twice should produce equivalent results."""
        result1 = pipeline_service.process_text(sample_resume_text)
        result2 = pipeline_service.process_text(sample_resume_text)

        assert result1.cleaned_text == result2.cleaned_text
        assert len(result1.sections) == len(result2.sections)
        assert len(result1.chunks) == len(result2.chunks)
        assert result1.word_count == result2.word_count
