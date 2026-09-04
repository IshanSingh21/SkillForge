"""
SkillForge AI — Roadmap & Interview Data Models.

Pydantic models for explainable, structured learning roadmaps and
interview question sets grounded in career knowledge retrieval.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.skillforge.models.rag import SourceCitation


# ── Learning Roadmap Enums ─────────────────────────────────────────────


class RoadmapPriority(str, Enum):
    """Priority tier for a roadmap skill item."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SkillGapStatus(str, Enum):
    """Status of the skill in the candidate's profile."""

    MISSING = "missing"
    PARTIAL = "partial"
    DEMONSTRATED = "demonstrated"


class RoadmapStageName(str, Enum):
    """Standardized roadmap stage names."""

    STAGE_1 = "Stage 1: Foundations & Critical Prerequisites"
    STAGE_2 = "Stage 2: Core Role Competencies"
    STAGE_3 = "Stage 3: Advanced Specialization & Production"


# ── Learning Resource Models ───────────────────────────────────────────


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
        default=ResourceType.DOCUMENTATION,
        description="Type of resource",
    )
    estimated_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated hours to complete",
    )
class RoadmapMilestone(BaseModel):
    """A single milestone in the learning roadmap (legacy alias)."""

    title: str = Field(..., description="Milestone title")
    description: str = Field(default="", description="What this milestone covers")
    skills_addressed: list[str] = Field(default_factory=list, description="Skills addressed")
    resources: list[LearningResource] = Field(default_factory=list, description="Recommended resources")
    estimated_weeks: float = Field(default=1.0, ge=0.0, description="Estimated weeks to complete")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1 = highest)")


# ── Structured Roadmap Skill Item & Stage Models ───────────────────────


class RoadmapSkillItem(BaseModel):
    """
    A single prioritized skill recommendation in the learning roadmap.

    Carries explainable priority score, prerequisite dependencies, learning objectives,
    and supporting citations retrieved from the career knowledge base.
    """

    skill: str = Field(..., description="Canonical name of the skill")
    category: str = Field(default="technical", description="Skill taxonomy category")
    priority: RoadmapPriority = Field(default=RoadmapPriority.HIGH, description="Priority tier")
    priority_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Explainable numerical score")
    priority_reason: str = Field(default="", description="Detailed explanation of why this skill is prioritized")
    status: SkillGapStatus = Field(default=SkillGapStatus.MISSING, description="Current gap status")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Candidate match similarity")
    prerequisites: list[str] = Field(default_factory=list, description="Required foundational skills")
    stage_number: int = Field(default=1, ge=1, le=3, description="Assigned roadmap stage (1, 2, or 3)")
    stage_name: str = Field(default=RoadmapStageName.STAGE_1.value, description="Display name of stage")
    relationship_to_role: str = Field(default="", description="How this skill connects to the target job")
    learning_objectives: list[str] = Field(default_factory=list, description="Core concepts and milestones to master")
    recommended_project: str = Field(default="", description="Hands-on portfolio project applying this skill")
    estimated_weeks: float = Field(default=2.0, ge=0.5, description="Estimated time in weeks to reach proficiency")
    supporting_citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Citations from career knowledge base grounding this recommendation",
    )
    has_supporting_knowledge: bool = Field(
        default=True,
        description="Whether this recommendation is backed by retrieved local knowledge base docs",
    )


class RoadmapStage(BaseModel):
    """A distinct stage in the learning roadmap organizing skills sequentially."""

    stage_number: int = Field(..., ge=1, le=3, description="Stage sequence number (1, 2, or 3)")
    stage_name: str = Field(..., description="Stage title")
    focus_area: str = Field(default="", description="Primary thematic objective of this stage")
    estimated_duration_weeks: float = Field(default=0.0, ge=0.0, description="Total duration for stage")
    skills: list[RoadmapSkillItem] = Field(default_factory=list, description="Ordered skills in this stage")


class LearningRoadmap(BaseModel):
    """
    Complete personalized learning roadmap generated from skill-gap analysis
    and grounded in the career knowledge base.
    """

    target_role: str = Field(..., description="Target job title or role")
    overall_match_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Resume match score from M5")
    summary: str = Field(default="", description="High-level executive summary of the learning plan")
    demonstrated_skills: list[str] = Field(
        default_factory=list,
        description="Skills already strongly demonstrated by candidate (excluded from learning roadmap)",
    )
    partially_matched_skills: list[str] = Field(
        default_factory=list,
        description="Skills candidate partially possesses needing enhancement",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Critical missing skills requiring dedicated learning",
    )
    stages: list[RoadmapStage] = Field(
        default_factory=list,
        description="Ordered learning stages (Stage 1 -> Stage 2 -> Stage 3)",
    )
    total_estimated_weeks: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated roadmap duration in weeks",
    )
    key_focus_areas: list[str] = Field(
        default_factory=list,
        description="Top 3 priority focus themes",
    )
    generated_with_llm: bool = Field(default=False, description="Whether LLM was used for summary synthesis")
    model_used: str = Field(default="deterministic-rule-engine", description="Model/engine identifier")


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
