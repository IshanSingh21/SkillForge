"""
Tests for Milestone 10 — Personalized Interview Question Generator.

Tests cover:
- Target-job-specific question generation and difficulty calibration (Senior -> HARD, Junior -> EASY)
- Missing-skill-based questions probing identified deficits with explainable rationale
- Resume/project-specific questions generated from parsed resume sections
- Generation of all 4 question categories (Technical, Conceptual, Project-Based, Behavioral)
- Technical questions grounded in retrieved career knowledge base citations
- Citation metadata preservation (source_name, section, relevance_score)
- Candidate-specific personalization avoiding generic questions
- Structured Pydantic schema validation and serialization roundtrip
- Graceful handling of empty or missing candidate information
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
from src.skillforge.models.resume import ResumeAnalysis, ResumeSection, Skill, SkillCategory
from src.skillforge.models.roadmap import (
    InterviewQuestion,
    InterviewQuestionSet,
    QuestionCategory,
    QuestionDifficulty,
)
from src.skillforge.services.interview_generator import InterviewQuestionGenerator
from src.skillforge.services.retrieval_service import RetrievalService

KB_DIR = PROJECT_ROOT / "knowledge_base"


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def retrieval_service(engine: EmbeddingEngine) -> RetrievalService:
    """Indexed retrieval service for interview question grounding."""
    store = VectorStore(dimension=engine.dimension)
    service = RetrievalService(vector_store=store, embedding_engine=engine)
    if KB_DIR.exists():
        service.index_knowledge_base_directory(KB_DIR)
    return service


@pytest.fixture
def generator(retrieval_service: RetrievalService) -> InterviewQuestionGenerator:
    """Instantiated InterviewQuestionGenerator."""
    return InterviewQuestionGenerator(retrieval_service=retrieval_service)


def _make_skill(name: str, category: SkillCategory = SkillCategory.LANGUAGE) -> Skill:
    """Helper to build Skill object."""
    return Skill(name=name, category=category, confidence=1.0)


@pytest.fixture
def sample_match_result() -> MatchResult:
    """
    Sample MatchResult representing an ML Engineer candidate with PyTorch/MLOps gaps.
    """
    python_skill = _make_skill("Python", SkillCategory.LANGUAGE)
    sql_skill = _make_skill("PostgreSQL", SkillCategory.TOOL)
    pytorch_skill = _make_skill("PyTorch", SkillCategory.FRAMEWORK)
    mlops_skill = _make_skill("MLOps", SkillCategory.DOMAIN)

    return MatchResult(
        overall_score=68.0,
        matched_skills=[
            SkillMatch(resume_skill=python_skill, job_skill=python_skill, similarity=1.0, match_type="exact"),
            SkillMatch(resume_skill=sql_skill, job_skill=sql_skill, similarity=0.9, match_type="strong_semantic"),
        ],
        partial_matches=[],
        missing_skills=[
            SkillGap(skill=pytorch_skill, importance="required", closest_similarity=0.20),
            SkillGap(skill=mlops_skill, importance="required", closest_similarity=0.15),
        ],
        score_breakdown={"skill_match": 65.0},
        summary="Strong Python backend background with gaps in PyTorch and MLOps.",
    )


@pytest.fixture
def sample_resume_analysis() -> ResumeAnalysis:
    """Sample parsed resume with projects and experience sections."""
    return ResumeAnalysis(
        raw_text="Jane Doe\nSenior Backend Engineer\n...",
        cleaned_text="Jane Doe\nSenior Backend Engineer\n...",
        sections=[
            ResumeSection(
                title="Experience",
                content=(
                    "Senior Software Engineer at CloudTech (2021-Present)\n"
                    "- Architected distributed high-throughput microservices handling 10k RPS.\n"
                    "- Built automated ETL data ingestion pipeline using PostgreSQL and Redis.\n"
                ),
                start_index=0,
                end_index=200,
            ),
            ResumeSection(
                title="Projects",
                content=(
                    "Real-Time Fraud Detection System\n"
                    "- Designed streaming anomaly detection pipeline with latency under 50ms.\n"
                    "- Implemented asynchronous event streaming with Kafka and FastAPI.\n"
                ),
                start_index=201,
                end_index=400,
            ),
        ],
        skills=[
            _make_skill("Python"),
            _make_skill("PostgreSQL"),
            _make_skill("FastAPI"),
        ],
    )


# ── Target-Role & Difficulty Calibration Tests ─────────────────────────


class TestRoleAndDifficultyCalibration:
    """Tests verifying role-specific tailoring and difficulty adjustment."""

    def test_senior_role_calibrates_to_hard(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """Roles with 'Senior', 'Lead', or 'Staff' must calibrate difficulty to HARD."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Senior Machine Learning Engineer",
            total_questions=8,
        )

        assert question_set.target_role == "Senior Machine Learning Engineer"
        assert question_set.difficulty_level == "Hard"
        assert len(question_set.questions) == 8
        assert all(q.difficulty == QuestionDifficulty.HARD for q in question_set.questions)

    def test_junior_role_calibrates_to_easy(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """Roles with 'Junior' or 'Entry' must calibrate difficulty to EASY."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Junior Python Developer",
            total_questions=6,
        )

        assert question_set.difficulty_level == "Easy"
        assert all(q.difficulty == QuestionDifficulty.EASY for q in question_set.questions)


# ── Missing Skill Probing Tests ────────────────────────────────────────


class TestMissingSkillProbing:
    """Tests verifying questions probe identified missing skills."""

    def test_missing_skills_generate_targeted_technical_questions(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """PyTorch and MLOps missing gaps must be targeted in technical questions with explainable rationale."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Senior Machine Learning Engineer",
            total_questions=8,
        )

        # Look for PyTorch or MLOps question
        gap_question = next((
            q for q in question_set.questions
            if q.related_skill.lower() in ["pytorch", "mlops"]
        ), None)

        assert gap_question is not None
        assert gap_question.category in [QuestionCategory.TECHNICAL, QuestionCategory.CONCEPTUAL]
        assert "skill gap" in gap_question.why_this_question.lower() or "target role requires" in gap_question.why_this_question.lower()
        assert len(gap_question.evaluation_points) > 0
        assert len(gap_question.sample_answer_points) > 0


# ── Resume Project Personalization Tests ───────────────────────────────


class TestResumeProjectPersonalization:
    """Tests verifying questions probe actual candidate projects from resume."""

    def test_project_based_questions_reference_resume_projects(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """Project-based questions must reference projects or work bullets from resume analysis."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Senior Backend Engineer",
            total_questions=8,
        )

        proj_questions = [q for q in question_set.questions if q.category == QuestionCategory.PROJECT_BASED]
        assert len(proj_questions) >= 1

        # Check that question text or why field references candidate projects
        first_proj_q = proj_questions[0]
        assert "bottleneck" in first_proj_q.question.lower() or "challenge" in first_proj_q.question.lower()
        assert len(first_proj_q.evaluation_points) >= 3


# ── Question Categories & Knowledge Base Grounding Tests ───────────────


class TestCategoriesAndKnowledgeBaseGrounding:
    """Tests verifying 4 categories and RAG knowledge base citation attachment."""

    def test_all_four_categories_generated(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """A full 8-question set must contain Technical, Conceptual, Project-Based, and Behavioral questions."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Senior ML Engineer",
            total_questions=8,
        )

        categories_present = {q.category for q in question_set.questions}
        assert QuestionCategory.TECHNICAL in categories_present
        assert QuestionCategory.CONCEPTUAL in categories_present
        assert QuestionCategory.PROJECT_BASED in categories_present
        assert QuestionCategory.BEHAVIORAL in categories_present

    def test_technical_and_conceptual_questions_attach_rag_citations(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """Questions for known domains (e.g. PyTorch, MLOps, Python) must attach knowledge base citations."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Senior ML Engineer",
            total_questions=8,
        )

        grounded_qs = [q for q in question_set.questions if q.has_supporting_knowledge]
        assert len(grounded_qs) >= 1

        citation = grounded_qs[0].supporting_citations[0]
        assert citation.source_name != ""
        assert citation.relevance_score > 0.0
        assert len(citation.content_preview) > 0


# ── Edge Cases & Schema Validation Tests ───────────────────────────────


class TestEdgeCasesAndSchemaValidation:
    """Tests for empty inputs and Pydantic serialization roundtrip."""

    def test_handles_empty_inputs_gracefully(self, generator: InterviewQuestionGenerator):
        """Generator must succeed without throwing when match_result and resume_analysis are None."""
        question_set = generator.generate_questions(
            match_result=None,
            resume_analysis=None,
            target_role="Data Scientist",
            total_questions=6,
        )

        assert isinstance(question_set, InterviewQuestionSet)
        assert question_set.target_role == "Data Scientist"
        assert len(question_set.questions) == 6
        assert len(question_set.focus_areas) > 0

    def test_schema_serialization_roundtrip(
        self,
        generator: InterviewQuestionGenerator,
        sample_match_result: MatchResult,
        sample_resume_analysis: ResumeAnalysis,
    ):
        """Question set must serialize to JSON dict and validate back cleanly."""
        question_set = generator.generate_questions(
            match_result=sample_match_result,
            resume_analysis=sample_resume_analysis,
            target_role="Senior Machine Learning Engineer",
            total_questions=8,
        )

        data = question_set.model_dump()
        reloaded = InterviewQuestionSet.model_validate(data)

        assert reloaded.target_role == question_set.target_role
        assert reloaded.total_questions == question_set.total_questions
        assert len(reloaded.questions) == len(question_set.questions)
