"""
SkillForge AI — Knowledge Base Loader & Document Parser.

Loads, parses, and chunks local markdown career knowledge base documents with
rich section hierarchy and metadata preservation for RAG retrieval.

Features:
    - Structured Markdown parsing with heading level detection (# Title, ## Section)
    - Metadata extraction (topic, doc_id, section, file_path, tags)
    - Section-aware chunking preserving section titles for citation grounding
    - Easily extensible for adding new career topics or external knowledge collections
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.skillforge.data.chunker import TextChunker
from src.skillforge.models.rag import CitationSource
from src.skillforge.models.resume import TextChunk
from src.skillforge.utils.exceptions import KnowledgeBaseError
from src.skillforge.utils.logging import logger


class DocumentSection(BaseModel):
    """A distinct structured section within a knowledge document."""

    heading: str = Field(..., description="Section heading title (e.g. 'Fundamental Concepts')")
    content: str = Field(..., description="Full text content of the section")
    level: int = Field(default=2, description="Markdown heading level (1 for #, 2 for ##, etc.)")


class KnowledgeDocument(BaseModel):
    """Structured representation of a parsed career knowledge base document."""

    doc_id: str = Field(..., description="Unique slug/identifier for the document")
    title: str = Field(..., description="Document display title")
    topic: str = Field(..., description="Primary topic category (e.g. 'Machine Learning')")
    file_path: str = Field(default="", description="Absolute or relative file path")
    sections: list[DocumentSection] = Field(default_factory=list, description="Parsed sections")
    raw_content: str = Field(default="", description="Original unparsed markdown text")
    tags: list[str] = Field(default_factory=list, description="Associated skill/topic tags")


class KnowledgeBaseLoader:
    """
    Loads, parses, and converts markdown knowledge base documents into
    searchable, section-tagged TextChunks for the vector store.
    """

    # Heading regex pattern for markdown
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def load_file(self, file_path: str | Path) -> KnowledgeDocument:
        """
        Parse a single markdown file into a structured KnowledgeDocument.

        Args:
            file_path: Path to the .md file.

        Returns:
            KnowledgeDocument instance with parsed title, topic, and sections.

        Raises:
            KnowledgeBaseError: If the file cannot be read or is invalid.
        """
        path = Path(file_path)
        if not path.exists():
            raise KnowledgeBaseError(f"Knowledge document not found at: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            doc_id = path.stem.lower()
            title, sections = self._parse_markdown_sections(content, fallback_title=path.stem)
            topic = self._derive_topic(doc_id, title)
            tags = self._generate_tags(topic, title)

            return KnowledgeDocument(
                doc_id=doc_id,
                title=title,
                topic=topic,
                file_path=str(path.resolve()),
                sections=sections,
                raw_content=content,
                tags=tags,
            )

        except Exception as e:
            logger.error("Failed to load knowledge document", file=str(path), error=str(e))
            raise KnowledgeBaseError(f"Error loading knowledge document {path.name}: {e}") from e

    def load_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
    ) -> list[KnowledgeDocument]:
        """
        Discover and load all markdown files from a directory.

        Args:
            directory_path: Path to knowledge_base folder.
            recursive: Whether to scan subdirectories.

        Returns:
            List of parsed KnowledgeDocument objects.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning("Knowledge base directory does not exist", path=str(dir_path))
            return []

        pattern = "**/*.md" if recursive else "*.md"
        md_files = sorted(list(dir_path.glob(pattern)))

        documents: list[KnowledgeDocument] = []
        for file_path in md_files:
            try:
                doc = self.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                logger.warning("Skipping invalid knowledge file", file=str(file_path), error=str(e))

        logger.info(
            "Loaded knowledge base documents",
            count=len(documents),
            directory=str(dir_path),
        )
        return documents

    def chunk_document(
        self,
        doc: KnowledgeDocument,
        chunker: TextChunker,
    ) -> list[TextChunk]:
        """
        Convert a KnowledgeDocument into section-tagged TextChunks using the provided TextChunker.

        Args:
            doc: Parsed KnowledgeDocument.
            chunker: Configured TextChunker instance.

        Returns:
            List of TextChunk models ready for vector indexing.
        """
        chunks: list[TextChunk] = []

        if not doc.sections:
            # Fallback if no markdown sections were parsed
            return chunker.chunk_text(
                text=doc.raw_content,
                source=CitationSource.KNOWLEDGE_BASE.value,
                source_name=doc.title,
                section="General Overview",
                metadata={
                    "topic": doc.topic,
                    "doc_id": doc.doc_id,
                    "file_path": doc.file_path,
                    "tags": doc.tags,
                },
            )

        for sec in doc.sections:
            sec_text = sec.content.strip()
            if not sec_text:
                continue

            sec_chunks = chunker.chunk_text(
                text=sec_text,
                source=CitationSource.KNOWLEDGE_BASE.value,
                source_name=doc.title,
                section=sec.heading,
                metadata={
                    "topic": doc.topic,
                    "doc_id": doc.doc_id,
                    "file_path": doc.file_path,
                    "tags": doc.tags,
                },
            )
            chunks.extend(sec_chunks)

        return chunks

    def chunk_directory(
        self,
        directory_path: str | Path,
        chunker: TextChunker,
        recursive: bool = True,
    ) -> list[TextChunk]:
        """Convenience method to load and chunk an entire directory of documents."""
        docs = self.load_directory(directory_path, recursive=recursive)
        all_chunks: list[TextChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc, chunker))
        return all_chunks

    # ── Parsing Helpers ────────────────────────────────────────────────

    def _parse_markdown_sections(
        self,
        content: str,
        fallback_title: str,
    ) -> tuple[str, list[DocumentSection]]:
        """Parse raw markdown content into title and list of DocumentSection objects."""
        lines = content.splitlines()
        title = fallback_title.replace("_", " ").title()
        sections: list[DocumentSection] = []

        current_heading = "Overview"
        current_level = 2
        current_lines: list[str] = []
        found_main_title = False

        for line in lines:
            match = self.HEADING_PATTERN.match(line)
            if match:
                hashes, heading_text = match.groups()
                level = len(hashes)

                # First H1 is taken as document title
                if level == 1 and not found_main_title:
                    title = heading_text.strip()
                    found_main_title = True
                    continue

                # Save previous section if it has content
                if current_lines:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            content="\n".join(current_lines).strip(),
                            level=current_level,
                        )
                    )
                    current_lines = []

                current_heading = heading_text.strip()
                current_level = level
            else:
                current_lines.append(line)

        # Append trailing section
        if current_lines:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    content="\n".join(current_lines).strip(),
                    level=current_level,
                )
            )

        return title, sections

    @staticmethod
    def _derive_topic(doc_id: str, title: str) -> str:
        """Derive a clean topic name from doc_id or title."""
        topic_map = {
            "machine_learning": "Machine Learning",
            "deep_learning": "Deep Learning",
            "natural_language_processing": "NLP",
            "computer_vision": "Computer Vision",
            "sql_and_databases": "SQL",
            "python_development": "Python",
            "mlops": "MLOps",
            "cloud_computing": "Cloud",
            "data_structures_and_algorithms": "Data Structures & Algorithms",
            "generative_ai": "Generative AI",
            "interview_preparation": "Interview Preparation",
            "skill_development": "Skill Development",
            "career_transitions": "Career Transitions",
            "industry_trends": "Industry Trends",
        }
        return topic_map.get(doc_id, title.split(":")[0].strip())

    @staticmethod
    def _generate_tags(topic: str, title: str) -> list[str]:
        """Generate search tags for a topic."""
        combined = f"{topic} {title}".lower()
        words = set(re.findall(r"\b[a-z]{3,}\b", combined))
        return sorted(list(words))
