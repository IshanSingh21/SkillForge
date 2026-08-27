"""
SkillForge AI — Document Chunking.

Splits text into overlapping chunks suitable for embedding and
vector indexing. Uses recursive character-based splitting to
respect sentence and paragraph boundaries where possible.

Usage:
    from src.skillforge.data.chunker import TextChunker

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_text(text, source="resume", source_name="resume.pdf")
"""

from __future__ import annotations

import uuid

from src.skillforge.models.resume import TextChunk
from src.skillforge.utils.exceptions import ChunkingError
from src.skillforge.utils.logging import logger


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.

    Uses a recursive splitting strategy that tries to split on:
    1. Double newlines (paragraph boundaries)
    2. Single newlines (line boundaries)
    3. Sentences (period + space)
    4. Spaces (word boundaries)
    5. Characters (last resort)

    This preserves semantic coherence within each chunk.
    """

    # Ordered list of separators to try, from most to least desirable
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
            separators: Custom separator priority list. Defaults to paragraph → sentence → word.

        Raises:
            ChunkingError: If configuration is invalid.
        """
        if chunk_size < 50:
            raise ChunkingError(
                f"chunk_size must be at least 50 (got {chunk_size})",
                detail="Very small chunks lose semantic meaning.",
            )
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})",
                detail="Overlap must be strictly less than chunk size.",
            )
        if chunk_overlap < 0:
            raise ChunkingError(
                f"chunk_overlap must be non-negative (got {chunk_overlap})",
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk_text(
        self,
        text: str,
        source: str = "resume",
        source_name: str = "",
        section: str = "",
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: The text to split into chunks.
            source: Source type ('resume', 'job_description', 'knowledge_base').
            source_name: Original document name or title.
            section: Resume section this text belongs to (if applicable).
            metadata: Additional metadata to attach to each chunk.

        Returns:
            List of TextChunk objects ready for embedding.

        Raises:
            ChunkingError: If chunking fails.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        try:
            raw_chunks = self._recursive_split(text.strip(), self.separators)

            # Apply overlap by merging with trailing context from previous chunk
            overlapped_chunks = self._apply_overlap(raw_chunks)

            # Build TextChunk objects with metadata
            result: list[TextChunk] = []
            for i, chunk_text in enumerate(overlapped_chunks):
                chunk = TextChunk(
                    chunk_id=f"{source}_{uuid.uuid4().hex[:8]}",
                    content=chunk_text,
                    source=source,
                    source_name=source_name,
                    section=section,
                    chunk_index=i,
                    metadata=metadata or {},
                )
                result.append(chunk)

            logger.info(
                "Text chunked",
                source=source,
                total_chars=len(text),
                num_chunks=len(result),
                avg_chunk_size=len(text) // max(len(result), 1),
            )

            return result

        except ChunkingError:
            raise
        except Exception as e:
            logger.error("Chunking failed", error=str(e))
            raise ChunkingError(
                f"Failed to chunk text: {e}",
                detail="An unexpected error occurred during text chunking.",
            ) from e

    def chunk_sections(
        self,
        sections: list[dict],
        source: str = "resume",
        source_name: str = "",
    ) -> list[TextChunk]:
        """
        Chunk multiple sections independently, preserving section metadata.

        Args:
            sections: List of dicts with 'title' and 'content' keys.
            source: Source type.
            source_name: Document name.

        Returns:
            Flat list of TextChunk objects from all sections.
        """
        all_chunks: list[TextChunk] = []

        for sec in sections:
            title = sec.get("title", "")
            content = sec.get("content", "")

            if not content.strip():
                continue

            chunks = self.chunk_text(
                text=content,
                source=source,
                source_name=source_name,
                section=title,
            )
            all_chunks.extend(chunks)

        return all_chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the separator priority list.

        Tries the first separator; if resulting pieces are still too large,
        recursively splits those pieces with the next separator.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator in priority order
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: character-level splitting
                return self._hard_split(text)

            if separator not in text:
                continue

            pieces = text.split(separator)
            chunks: list[str] = []
            current = ""

            for piece in pieces:
                # Would adding this piece exceed the chunk size?
                candidate = (
                    current + separator + piece if current else piece
                )

                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    # Save the current chunk (if non-empty)
                    if current.strip():
                        chunks.append(current.strip())

                    # If the piece itself is too large, recursively split it
                    if len(piece) > self.chunk_size:
                        sub_chunks = self._recursive_split(
                            piece, separators[i + 1:]
                        )
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = piece

            # Don't forget the last accumulated piece
            if current.strip():
                chunks.append(current.strip())

            if chunks:
                return chunks

        # Fallback: hard character split
        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        """Split text into fixed-size chunks as a last resort."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i: i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply overlap between consecutive chunks.

        Each chunk (except the first) is prepended with the last
        `chunk_overlap` characters from the previous chunk to
        maintain context continuity.
        """
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-self.chunk_overlap:]

            # Try to start overlap at a word boundary
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]

            overlapped = f"...{overlap_text} {chunks[i]}"
            result.append(overlapped)

        return result
