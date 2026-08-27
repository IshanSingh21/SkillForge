"""
SkillForge AI — Roadmap & Interview Data Models.

Pydantic models for LLM-generated learning roadmaps and
interview question sets.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Learning Roadmap Models ───────────────────────────────────────────


class ResourceType(str, Enum):
    """Type of learning resource."""

    COURSE = "course"
    TUTORIAL = "tutorial"
    BOOK = "book"
    DOCUMENTATION = "documentation"
    PROJECT = "project"
    CERTIFICATION = "certification"
    VIDEO = "video"
    ARTICLE = "article"


class LearningResource(BaseModel):
    """A specific learning resource recommended for a skill."""

    title: str = Field(..., description="Resource title")
    url: str = Field(default="", description="Link to the resource")
    resource_type: ResourceType = Field(
        default=ResourceType.COURSE,
        description="Type of resource",
    )
    estimated_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated hours to complete",
    )
    is_free: bool = Field(default=True, description="Whether the resource is free")


class RoadmapMilestone(BaseModel):
    """A single milestone in the learning roadmap."""

    title: str = Field(..., description="Milestone title (e.g., 'Learn Docker Basics')")
    description: str = Field(default="", description="What this milestone covers")
    skills_addressed: list[str] = Field(
        default_factory=list,
        description="Which skill gaps this milestone addresses",
    )
    resources: list[LearningResource] = Field(
        default_factory=list,
        description="Recommended resources for this milestone",
    )
    estimated_weeks: float = Field(
        default=1.0,
        ge=0.0,
        description="Estimated weeks to complete",
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Priority level (1 = highest)",
    )


class LearningRoadmap(BaseModel):
    """
    A personalized learning roadmap generated from skill-gap analysis.

    Contains ordered milestones with resources, time estimates,
    and a summary of the overall development plan.
    """

    target_role: str = Field(..., description="The target job role")
    milestones: list[RoadmapMilestone] = Field(
        default_factory=list,
        description="Ordered learning milestones",
    )
    total_estimated_weeks: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated time to complete the roadmap",
    )
    summary: str = Field(
        default="",
        description="High-level summary of the learning plan",
    )
    key_focus_areas: list[str] = Field(
        default_factory=list,
        description="Top priority areas to focus on",
    )


# ── Interview Question Models ─────────────────────────────────────────


class QuestionDifficulty(str, Enum):
    """Difficulty level of an interview question."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionCategory(str, Enum):
    """Category of interview question."""

    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    PROBLEM_SOLVING = "problem_solving"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CULTURE_FIT = "culture_fit"


class InterviewQuestion(BaseModel):
    """A single generated interview question with guidance."""

    question: str = Field(..., description="The interview question")
    category: QuestionCategory = Field(
        default=QuestionCategory.TECHNICAL,
        description="Question category",
    )
    difficulty: QuestionDifficulty = Field(
        default=QuestionDifficulty.MEDIUM,
        description="Difficulty level",
    )
    related_skill: str = Field(
        default="",
        description="The skill this question targets",
    )
    guidance: str = Field(
        default="",
        description="Tips for answering this question well",
    )
    sample_answer_points: list[str] = Field(
        default_factory=list,
        description="Key points a strong answer should cover",
    )


class InterviewQuestionSet(BaseModel):
    """A set of interview questions generated for a target role."""

    target_role: str = Field(..., description="The target job role")
    questions: list[InterviewQuestion] = Field(
        default_factory=list,
        description="Generated interview questions",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Key areas the questions cover",
    )
    preparation_tips: list[str] = Field(
        default_factory=list,
        description="General interview preparation advice",
    )
