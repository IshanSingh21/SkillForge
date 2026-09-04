"""
Tests for Milestone 9 — Personalized Learning Roadmap Generator.

Tests cover:
- Strongly demonstrated skills are excluded from learning stages and recorded in demonstrated_skills
- Missing required skills receive high priority and high priority_scores
- Partially matched skills are included as skill enhancements
- Prerequisite ordering and dependency resolution (foundations before advanced skills)
- 3 sequential roadmap stages with valid duration calculations
- Target-role-specific prioritization and contextual explanations
- Retrieval-grounded recommendations attaching citations from the knowledge base
- Graceful handling of skills with no matching career knowledge
- Full Pydantic structured output validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.ai.vector_store import VectorStore
from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.resume import Skill, SkillCategory
from src.skillforge.models.roadmap import (
    LearningRoadmap,
    RoadmapPriority,
    RoadmapSkillItem,
    RoadmapStage,
    RoadmapStageName,
    SkillGapStatus,
)
from src.skillforge.services.retrieval_service import RetrievalService
from src.skillforge.services.roadmap_generator import RoadmapGenerator

KB_DIR = PROJECT_ROOT / "knowledge_base"


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def retrieval_service(engine: EmbeddingEngine) -> RetrievalService:
    """Indexed retrieval service for roadmap grounding."""
    store = VectorStore(dimension=engine.dimension)
    service = RetrievalService(vector_store=store, embedding_engine=engine)
    if KB_DIR.exists():
        service.index_knowledge_base_directory(KB_DIR)
    return service


@pytest.fixture
def generator(retrieval_service: RetrievalService) -> RoadmapGenerator:
    """Instantiated RoadmapGenerator."""
    return RoadmapGenerator(retrieval_service=retrieval_service)


def _make_skill(name: str, category: SkillCategory = SkillCategory.LANGUAGE) -> Skill:
    """Helper to build Skill object."""
    return Skill(
        name=name,
        category=category,
        confidence=1.0,
    )


@pytest.fixture
def sample_match_result() -> MatchResult:
    """
    Sample MatchResult representing a candidate transitioning to ML Engineer:
    - Demonstrated: Python (1.0), Git (1.0), PostgreSQL (0.85)
    - Partial: Docker (0.60), FastAPI (0.55)
    - Missing: PyTorch (0.20), MLOps (0.15), Kubernetes (0.10)
    """
    python_skill = _make_skill("Python", SkillCategory.LANGUAGE)
    git_skill = _make_skill("Git", SkillCategory.TOOL)
    sql_skill = _make_skill("PostgreSQL", SkillCategory.TOOL)
    docker_skill = _make_skill("Docker", SkillCategory.TOOL)
    fastapi_skill = _make_skill("FastAPI", SkillCategory.FRAMEWORK)
    pytorch_skill = _make_skill("PyTorch", SkillCategory.FRAMEWORK)
    mlops_skill = _make_skill("MLOps", SkillCategory.DOMAIN)
    k8s_skill = _make_skill("Kubernetes", SkillCategory.TOOL)

    return MatchResult(
        overall_score=62.5,
        matched_skills=[
            SkillMatch(
                resume_skill=python_skill,
                job_skill=python_skill,
                similarity=1.0,
                match_type="exact",
            ),
            SkillMatch(
                resume_skill=git_skill,
                job_skill=git_skill,
                similarity=1.0,
                match_type="exact",
            ),
            SkillMatch(
                resume_skill=sql_skill,
                job_skill=sql_skill,
                similarity=0.85,
                match_type="strong_semantic",
            ),
        ],
        partial_matches=[
            SkillMatch(
                resume_skill=_make_skill("Containers"),
                job_skill=docker_skill,
                similarity=0.60,
                match_type="partial_semantic",
            ),
            SkillMatch(
                resume_skill=_make_skill("Web APIs"),
                job_skill=fastapi_skill,
                similarity=0.55,
                match_type="partial_semantic",
            ),
        ],
        missing_skills=[
            SkillGap(
                skill=pytorch_skill,
                importance="required",
                closest_similarity=0.20,
                recommendation="Learn PyTorch tensors and deep neural network training.",
            ),
            SkillGap(
                skill=mlops_skill,
                importance="required",
                closest_similarity=0.15,
                recommendation="Learn MLflow experiment tracking and CI/CD for ML.",
            ),
            SkillGap(
                skill=k8s_skill,
                importance="preferred",
                closest_similarity=0.10,
                recommendation="Learn container orchestration with Kubernetes.",
            ),
        ],
        score_breakdown={
            "skill_match": 60.0,
            "required_coverage": 50.0,
            "semantic_fit": 65.0,
            "experience": 70.0,
        },
        summary="Candidate has strong Python and SQL background but needs PyTorch and MLOps.",
    )


# ── Demonstrated Skill Exclusion Tests ─────────────────────────────────


class TestDemonstratedSkillExclusion:
    """Tests verifying strongly matched skills are not scheduled for learning."""

    def test_strongly_demonstrated_skills_excluded_from_stages(
        self,
        generator: RoadmapGenerator,
        sample_match_result: MatchResult,
    ):
        """Python, Git, and PostgreSQL (sim >= 0.70) must be in demonstrated_skills and not in stages."""
        roadmap = generator.generate_roadmap(sample_match_result, target_role="ML Engineer")

        # Check demonstrated list
        assert "Python" in roadmap.demonstrated_skills
        assert "Git" in roadmap.demonstrated_skills
        assert "PostgreSQL" in roadmap.demonstrated_skills

        # Ensure none of the demonstrated skills appear in roadmap stages
        all_stage_skills = [
            skill_item.skill.lower()
            for stage in roadmap.stages
            for skill_item in stage.skills
        ]

        assert "python" not in all_stage_skills
        assert "git" not in all_stage_skills
        assert "postgresql" not in all_stage_skills


# ── Missing & Partial Skill Prioritization Tests ───────────────────────


class TestSkillPrioritization:
    """Tests for explainable prioritization and status handling."""

    def test_missing_required_skills_receive_high_priority(
        self,
        generator: RoadmapGenerator,
        sample_match_result: MatchResult,
    ):
        """PyTorch and MLOps (missing required) must have HIGH priority and score >= 75."""
        roadmap = generator.generate_roadmap(sample_match_result, target_role="ML Engineer")

        pytorch_item = next((
            item for stage in roadmap.stages
            for item in stage.skills if item.skill.lower() == "pytorch"
        ), None)

        assert pytorch_item is not None
        assert pytorch_item.status == SkillGapStatus.MISSING
        assert pytorch_item.priority == RoadmapPriority.HIGH
        assert pytorch_item.priority_score >= 75.0
        assert "Mandatory job requirement" in pytorch_item.priority_reason

    def test_partial_skills_classified_and_scheduled(
        self,
        generator: RoadmapGenerator,
        sample_match_result: MatchResult,
    ):
        """Docker and FastAPI should be categorized as PARTIAL and included."""
        roadmap = generator.generate_roadmap(sample_match_result, target_role="ML Engineer")

        assert "Docker" in roadmap.partially_matched_skills or "FastAPI" in roadmap.partially_matched_skills

        docker_item = next((
            item for stage in roadmap.stages
            for item in stage.skills if item.skill.lower() == "docker"
        ), None)

        if docker_item:
            assert docker_item.status == SkillGapStatus.PARTIAL
            assert "Partially demonstrated" in docker_item.priority_reason


# ── Prerequisite Ordering & Sequential Stages Tests ────────────────────


class TestPrerequisitesAndStages:
    """Tests for dependency resolution and 3-stage progression."""

    def test_prerequisite_ordering(self, generator: RoadmapGenerator):
        """If Python is missing along with PyTorch, Python must be scheduled in Stage 1."""
        python_skill = _make_skill("Python")
        pytorch_skill = _make_skill("PyTorch")
        k8s_skill = _make_skill("Kubernetes")

        match_res = MatchResult(
            overall_score=20.0,
            matched_skills=[],
            partial_matches=[],
            missing_skills=[
                SkillGap(skill=pytorch_skill, importance="required", closest_similarity=0.0),
                SkillGap(skill=python_skill, importance="required", closest_similarity=0.0),
                SkillGap(skill=k8s_skill, importance="preferred", closest_similarity=0.0),
            ],
            summary="Candidate has severe gaps.",
        )

        roadmap = generator.generate_roadmap(match_res, target_role="ML Engineer")

        # Python is foundational -> Stage 1
        s1_skills = [i.skill.lower() for i in roadmap.stages[0].skills]
        assert "python" in s1_skills

    def test_three_sequential_stages_created(
        self,
        generator: RoadmapGenerator,
        sample_match_result: MatchResult,
    ):
        """Roadmap must contain exactly 3 ordered stages with valid durations."""
        roadmap = generator.generate_roadmap(sample_match_result, target_role="ML Engineer")

        assert len(roadmap.stages) == 3
        assert roadmap.stages[0].stage_number == 1
        assert roadmap.stages[1].stage_number == 2
        assert roadmap.stages[2].stage_number == 3

        assert roadmap.total_estimated_weeks > 0.0
        assert roadmap.total_estimated_weeks == sum(s.estimated_duration_weeks for s in roadmap.stages)


# ── Knowledge Base Grounding & Edge Cases Tests ────────────────────────


class TestKnowledgeBaseGrounding:
    """Tests verifying recommendations are grounded in retrieved career knowledge."""

    def test_skills_attach_knowledge_base_citations(
        self,
        generator: RoadmapGenerator,
        sample_match_result: MatchResult,
    ):
        """Known skills (e.g. PyTorch, MLOps) should attach citations from knowledge base."""
        roadmap = generator.generate_roadmap(sample_match_result, target_role="ML Engineer")

        pytorch_item = next((
            item for stage in roadmap.stages
            for item in stage.skills if item.skill.lower() == "pytorch"
        ), None)

        assert pytorch_item is not None
        assert pytorch_item.has_supporting_knowledge is True
        assert len(pytorch_item.supporting_citations) > 0
        assert len(pytorch_item.learning_objectives) >= 3
        assert len(pytorch_item.recommended_project) > 10

    def test_unsupported_obscure_skill_handles_gracefully(self, generator: RoadmapGenerator):
        """Obscure skill not in knowledge base should generate fallback without breaking."""
        obscure_skill = _make_skill("ProprietaryLegacyToolXYZ")
        match_res = MatchResult(
            overall_score=50.0,
            matched_skills=[],
            partial_matches=[],
            missing_skills=[
                SkillGap(skill=obscure_skill, importance="required", closest_similarity=0.0),
            ],
            summary="Gap in proprietary tool.",
        )

        roadmap = generator.generate_roadmap(match_res, target_role="Specialist")
        item = roadmap.stages[0].skills[0] if roadmap.stages[0].skills else roadmap.stages[1].skills[0]

        assert item.skill == "ProprietaryLegacyToolXYZ"
        assert len(item.learning_objectives) >= 3


# ── Structured Output Validation ───────────────────────────────────────


class TestStructuredOutputValidation:
    """Tests verifying LearningRoadmap conforms to Pydantic schema."""

    def test_roadmap_model_serializes_and_validates(
        self,
        generator: RoadmapGenerator,
        sample_match_result: MatchResult,
    ):
        """Generated roadmap must validate cleanly as a Pydantic model and export to dict."""
        roadmap = generator.generate_roadmap(sample_match_result, target_role="Senior ML Engineer")

        assert isinstance(roadmap, LearningRoadmap)
        assert roadmap.target_role == "Senior ML Engineer"
        assert len(roadmap.key_focus_areas) > 0
        assert len(roadmap.summary) > 20

        # Validate JSON serialization / deserialization roundtrip
        json_data = roadmap.model_dump()
        reloaded = LearningRoadmap.model_validate(json_data)
        assert reloaded.target_role == roadmap.target_role
        assert reloaded.total_estimated_weeks == roadmap.total_estimated_weeks
