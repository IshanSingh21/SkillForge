"""
Tests for src.skillforge.data.chunker — Text chunking.

Tests cover:
- Basic chunking with size and overlap
- Recursive splitting behavior
- Section-aware chunking
- Edge cases (empty input, very small text, single chunk)
- Chunk metadata
"""

from __future__ import annotations

import pytest

from src.skillforge.data.chunker import TextChunker
from src.skillforge.models.resume import TextChunk
from src.skillforge.utils.exceptions import ChunkingError


class TestTextChunker:
    """Tests for TextChunker.chunk_text()."""

    def test_short_text_produces_single_chunk(self, chunker: TextChunker):
        """Text shorter than chunk_size should produce exactly one chunk."""
        text = "A short resume summary."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_long_text_produces_multiple_chunks(self, small_chunker: TextChunker):
        """Text longer than chunk_size should be split into multiple chunks."""
        text = "This is a sentence. " * 50  # ~1000 chars
        chunks = small_chunker.chunk_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk's core content (minus overlap prefix) should be reasonable
            assert len(chunk.content) > 0

    def test_chunks_have_unique_ids(self, small_chunker: TextChunker):
        """Every chunk should have a unique chunk_id."""
        text = "Paragraph one content. " * 20 + "\n\n" + "Paragraph two content. " * 20
        chunks = small_chunker.chunk_text(text)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_chunks_have_correct_metadata(self, chunker: TextChunker):
        """Chunks should carry the source metadata passed to chunk_text()."""
        chunks = chunker.chunk_text(
            "Some content here.",
            source="resume",
            source_name="test_resume.pdf",
            section="Experience",
        )

        assert len(chunks) >= 1
        assert chunks[0].source == "resume"
        assert chunks[0].source_name == "test_resume.pdf"
        assert chunks[0].section == "Experience"

    def test_chunks_are_ordered(self, small_chunker: TextChunker):
        """Chunk indices should be sequential."""
        text = "Some content repeated many times. " * 30
        chunks = small_chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_overlap_is_applied(self, small_chunker: TextChunker):
        """Chunks after the first should contain overlap markers."""
        text = "First paragraph content here. " * 10 + "\n\n" + "Second paragraph. " * 10
        chunks = small_chunker.chunk_text(text)

        if len(chunks) > 1:
            # Second chunk and beyond should have overlap prefix
            assert chunks[1].content.startswith("...")

    def test_empty_text_returns_empty_list(self, chunker: TextChunker):
        """Empty or whitespace-only text should return an empty list."""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_returns_text_chunk_objects(self, chunker: TextChunker):
        """All returned items should be TextChunk instances."""
        chunks = chunker.chunk_text("Some resume content here.")
        assert all(isinstance(c, TextChunk) for c in chunks)


class TestTextChunkerConfiguration:
    """Tests for TextChunker initialization and configuration."""

    def test_rejects_tiny_chunk_size(self):
        """Chunk size below 50 should be rejected."""
        with pytest.raises(ChunkingError, match="at least 50"):
            TextChunker(chunk_size=10)

    def test_rejects_overlap_exceeding_chunk_size(self):
        """Overlap >= chunk_size should be rejected."""
        with pytest.raises(ChunkingError, match="less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ChunkingError, match="less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_rejects_negative_overlap(self):
        """Negative overlap should be rejected."""
        with pytest.raises(ChunkingError, match="non-negative"):
            TextChunker(chunk_size=100, chunk_overlap=-1)

    def test_zero_overlap_is_valid(self):
        """Zero overlap should be accepted and produce non-overlapping chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk_text("Word " * 100)

        if len(chunks) > 1:
            # No overlap marker on second chunk
            assert not chunks[1].content.startswith("...")


class TestChunkSections:
    """Tests for TextChunker.chunk_sections()."""

    def test_chunks_multiple_sections(self, small_chunker: TextChunker):
        """Multiple sections should each be chunked independently."""
        sections = [
            {"title": "Experience", "content": "Worked at company. " * 20},
            {"title": "Education", "content": "Studied at university. " * 20},
        ]
        chunks = small_chunker.chunk_sections(sections, source="resume")

        assert len(chunks) > 2  # Each section should produce multiple chunks
        # Should have chunks from both sections
        section_labels = set(c.section for c in chunks)
        assert "Experience" in section_labels
        assert "Education" in section_labels

    def test_skips_empty_sections(self, chunker: TextChunker):
        """Sections with empty content should be skipped."""
        sections = [
            {"title": "Experience", "content": "Real content here."},
            {"title": "Empty", "content": ""},
            {"title": "Also Empty", "content": "   "},
        ]
        chunks = chunker.chunk_sections(sections)

        assert len(chunks) == 1
        assert chunks[0].section == "Experience"
