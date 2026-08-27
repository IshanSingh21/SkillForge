"""
SkillForge AI — Resume Data Models.

Pydantic models representing the structured output of resume processing:
parsed text, extracted sections, identified skills, and text chunks
ready for embedding.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SkillCategory(str, Enum):
    """Broad categories for classifying extracted skills."""

    TECHNICAL = "technical"
    SOFT = "soft"
    TOOL = "tool"
    FRAMEWORK = "framework"
    LANGUAGE = "language"
    CERTIFICATION = "certification"
    DOMAIN = "domain"
    OTHER = "other"


class Skill(BaseModel):
    """A single skill extracted from text."""

    name: str = Field(..., description="Canonical skill name (e.g., 'Python')")
    category: SkillCategory = Field(
        default=SkillCategory.OTHER,
        description="Broad category of the skill",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from extraction (0.0–1.0)",
    )
    source: str = Field(
        default="",
        description="Where this skill was found (e.g., 'resume', 'job_description')",
    )

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Skill):
            return self.name.lower() == other.name.lower()
        return NotImplemented


class ResumeSection(BaseModel):
    """A detected section within a resume (e.g., Education, Experience)."""

    title: str = Field(..., description="Section heading (e.g., 'Work Experience')")
    content: str = Field(..., description="Full text content of the section")
    start_index: int = Field(default=0, description="Character offset in the original text")
    end_index: int = Field(default=0, description="End character offset")


class TextChunk(BaseModel):
    """
    A chunk of text prepared for embedding and vector indexing.

    Each chunk carries metadata about its source so RAG responses
    can provide accurate source citations.
    """

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    content: str = Field(..., description="The text content of the chunk")
    source: str = Field(
        default="resume",
        description="Source document type: 'resume', 'job_description', 'knowledge_base'",
    )
    source_name: str = Field(
        default="",
        description="Original filename or document title",
    )
    section: str = Field(
        default="",
        description="Resume section this chunk belongs to (if applicable)",
    )
    chunk_index: int = Field(default=0, description="Ordinal position within the source document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ResumeAnalysis(BaseModel):
    """
    Complete output of the resume processing pipeline.

    Encapsulates the raw text, cleaned text, detected sections,
    extracted skills, and generated text chunks.
    """

    raw_text: str = Field(..., description="Raw text extracted from the PDF")
    cleaned_text: str = Field(..., description="Preprocessed and cleaned text")
    sections: list[ResumeSection] = Field(
        default_factory=list,
        description="Detected resume sections",
    )
    skills: list[Skill] = Field(
        default_factory=list,
        description="Skills extracted from the resume",
    )
    chunks: list[TextChunk] = Field(
        default_factory=list,
        description="Text chunks ready for embedding",
    )
    filename: str = Field(default="", description="Original PDF filename")
    page_count: int = Field(default=0, description="Number of pages in the PDF")
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of processing",
    )
    processing_errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or issues encountered during processing",
    )

    @property
    def word_count(self) -> int:
        """Return the approximate word count of the cleaned text."""
        return len(self.cleaned_text.split())

    @property
    def has_sections(self) -> bool:
        """Return True if structured sections were detected."""
        return len(self.sections) > 0
