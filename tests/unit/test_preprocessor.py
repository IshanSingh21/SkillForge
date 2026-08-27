"""
Tests for src.skillforge.data.preprocessor — Text cleaning & section extraction.

Tests cover:
- Text cleaning (whitespace, encoding artifacts, Unicode)
- Section extraction (common resume headings)
- Edge cases (empty input, no sections, unusual formatting)
"""

from __future__ import annotations

import pytest

from src.skillforge.data.preprocessor import TextPreprocessor
from src.skillforge.utils.exceptions import PreprocessingError


class TestTextCleaning:
    """Tests for TextPreprocessor.clean_text()."""

    def test_removes_excessive_whitespace(self, preprocessor: TextPreprocessor):
        """Multiple spaces should be collapsed to one."""
        text = "Python    JavaScript     React"
        cleaned = preprocessor.clean_text(text)
        assert "    " not in cleaned
        assert "Python" in cleaned and "React" in cleaned

    def test_normalizes_line_endings(self, preprocessor: TextPreprocessor):
        """CRLF should be normalized to LF."""
        text = "Line 1\r\nLine 2\r\nLine 3"
        cleaned = preprocessor.clean_text(text)
        assert "\r\n" not in cleaned
        assert "Line 1" in cleaned

    def test_reduces_excessive_blank_lines(self, preprocessor: TextPreprocessor):
        """More than 2 consecutive blank lines should be reduced."""
        text = "Section 1\n\n\n\n\n\nSection 2"
        cleaned = preprocessor.clean_text(text)
        assert "\n\n\n\n" not in cleaned
        assert "Section 1" in cleaned and "Section 2" in cleaned

    def test_fixes_encoding_artifacts(self, preprocessor: TextPreprocessor):
        """Smart quotes and special characters should be normalized."""
        text = "\u201cHello\u201d \u2013 \u2018World\u2019"
        cleaned = preprocessor.clean_text(text)
        assert "\u201c" not in cleaned  # No smart quotes
        assert "Hello" in cleaned and "World" in cleaned

    def test_strips_line_whitespace(self, preprocessor: TextPreprocessor):
        """Each line should be stripped of leading/trailing whitespace."""
        text = "  Line 1  \n  Line 2  \n  Line 3  "
        cleaned = preprocessor.clean_text(text)
        lines = cleaned.split("\n")
        for line in lines:
            assert line == line.strip()

    def test_rejects_empty_input(self, preprocessor: TextPreprocessor):
        """Empty or whitespace-only input should raise PreprocessingError."""
        with pytest.raises(PreprocessingError, match="empty"):
            preprocessor.clean_text("")

        with pytest.raises(PreprocessingError, match="empty"):
            preprocessor.clean_text("   \n\n   ")

    def test_preserves_meaningful_content(self, preprocessor: TextPreprocessor):
        """Cleaning should not remove meaningful resume content."""
        text = "Python 3.10+ | FastAPI | Docker\nExperience: 5 years"
        cleaned = preprocessor.clean_text(text)
        assert "Python 3.10+" in cleaned
        assert "FastAPI" in cleaned
        assert "5 years" in cleaned


class TestSectionExtraction:
    """Tests for TextPreprocessor.extract_sections()."""

    def test_extracts_standard_sections(self, preprocessor: TextPreprocessor, sample_resume_text: str):
        """Standard resume sections should be detected."""
        cleaned = preprocessor.clean_text(sample_resume_text)
        sections = preprocessor.extract_sections(cleaned)

        assert len(sections) > 0
        titles = [s.title.lower() for s in sections]

        # Should find at least some of these common sections
        found_expected = any(
            keyword in title
            for title in titles
            for keyword in ["experience", "education", "skills", "summary"]
        )
        assert found_expected, f"Expected standard sections, got: {titles}"

    def test_sections_have_content(self, preprocessor: TextPreprocessor, sample_resume_text: str):
        """Each extracted section should have non-empty content."""
        cleaned = preprocessor.clean_text(sample_resume_text)
        sections = preprocessor.extract_sections(cleaned)

        for section in sections:
            assert section.content.strip(), f"Section '{section.title}' has no content"

    def test_sections_are_ordered(self, preprocessor: TextPreprocessor, sample_resume_text: str):
        """Sections should be ordered by their position in the text."""
        cleaned = preprocessor.clean_text(sample_resume_text)
        sections = preprocessor.extract_sections(cleaned)

        for i in range(1, len(sections)):
            assert sections[i].start_index >= sections[i - 1].start_index

    def test_no_sections_in_plain_text(self, preprocessor: TextPreprocessor):
        """Text without section headings should return an empty list."""
        text = "Just a plain paragraph with no section headings at all."
        sections = preprocessor.extract_sections(text)
        assert sections == []

    def test_empty_input_returns_empty(self, preprocessor: TextPreprocessor):
        """Empty input should return an empty list, not raise."""
        assert preprocessor.extract_sections("") == []
        assert preprocessor.extract_sections("   ") == []
