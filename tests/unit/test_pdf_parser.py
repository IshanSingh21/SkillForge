"""
Tests for src.skillforge.data.pdf_parser — PDF text extraction.

Tests cover:
- Successful extraction from valid PDFs
- Input validation (empty bytes, too large, invalid format)
- Metadata extraction
- Error handling for corrupted files
"""

from __future__ import annotations

import pytest

from src.skillforge.data.pdf_parser import PDFParser, PDFExtractionResult
from src.skillforge.utils.exceptions import PDFParsingError


class TestPDFParser:
    """Tests for PDFParser.extract_text()."""

    def test_extract_text_from_valid_pdf(self, pdf_parser: PDFParser, minimal_pdf_bytes: bytes):
        """A minimal valid PDF should produce a non-empty extraction result."""
        result = pdf_parser.extract_text(minimal_pdf_bytes)

        assert isinstance(result, PDFExtractionResult)
        assert result.page_count >= 1
        assert len(result.text.strip()) > 0
        assert "SkillForge" in result.text or "Hello" in result.text

    def test_extract_text_returns_page_list(self, pdf_parser: PDFParser, minimal_pdf_bytes: bytes):
        """The result should contain per-page text."""
        result = pdf_parser.extract_text(minimal_pdf_bytes)

        assert len(result.pages) == result.page_count
        assert isinstance(result.pages[0], str)

    def test_extract_text_includes_metadata(self, pdf_parser: PDFParser, minimal_pdf_bytes: bytes):
        """The result should include PDF metadata."""
        result = pdf_parser.extract_text(minimal_pdf_bytes)

        assert isinstance(result.metadata, dict)
        assert "page_count" in result.metadata

    def test_rejects_empty_bytes(self, pdf_parser: PDFParser):
        """Empty input should raise PDFParsingError."""
        with pytest.raises(PDFParsingError, match="Empty input"):
            pdf_parser.extract_text(b"")

    def test_rejects_non_pdf_bytes(self, pdf_parser: PDFParser):
        """Non-PDF bytes should be rejected based on magic bytes check."""
        with pytest.raises(PDFParsingError, match="valid PDF"):
            pdf_parser.extract_text(b"This is not a PDF file at all")

    def test_rejects_oversized_file(self, pdf_parser: PDFParser):
        """Files exceeding the size limit should be rejected."""
        # Create bytes just over the 20 MB limit
        oversized = b"%PDF-" + b"x" * (PDFParser.MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(PDFParsingError, match="too large"):
            pdf_parser.extract_text(oversized)

    def test_extract_text_from_path_file_not_found(self, pdf_parser: PDFParser):
        """Missing file path should raise PDFParsingError."""
        with pytest.raises(PDFParsingError, match="not found"):
            pdf_parser.extract_text_from_path("/nonexistent/path/resume.pdf")

    def test_error_has_detail_message(self, pdf_parser: PDFParser):
        """PDFParsingError should carry a helpful detail message."""
        with pytest.raises(PDFParsingError) as exc_info:
            pdf_parser.extract_text(b"")

        assert exc_info.value.detail != ""
