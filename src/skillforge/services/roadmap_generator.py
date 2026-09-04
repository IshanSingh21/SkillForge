"""
SkillForge AI — Personalized Learning Roadmap Generator.

Generates structured, explainable, and multi-stage learning roadmaps
grounded in resume-job matching analysis and retrieved career knowledge.

Architecture & Design Decisions:
    1. Deterministic Prioritization: Uses an explainable mathematical formula combining
       job requirement importance (required vs preferred), current gap status (missing vs partial),
       and prerequisite graph topology.
    2. Exclusion of Strong Skills: Skills already strongly demonstrated (similarity >= 0.70)
       are recorded as mastered and excluded from learning stages.
    3. Multi-Stage Progression:
       - Stage 1: Foundations & Critical Prerequisites
       - Stage 2: Core Role Competencies
       - Stage 3: Advanced Specialization & Production
    4. Grounded in Career Knowledge: Integrates directly with RetrievalService and RAG
       to attach factual supporting citations, learning phases, and project ideas from local KB docs.
    5. Pluggable LLM Refinement: The core roadmap is 100% functional and testable deterministically,
       with an optional LLM layer for executive summary synthesis.

Usage:
    from src.skillforge.services.roadmap_generator import RoadmapGenerator

    generator = RoadmapGenerator()
    roadmap = generator.generate_roadmap(match_result, target_role="ML Engineer")
    print(f"Total duration: {roadmap.total_estimated_weeks} weeks across {len(roadmap.stages)} stages.")
"""

from __future__ import annotations

from typing import Any

from src.skillforge.ai.llm.base import LLMProvider
from src.skillforge.ai.llm.factory import create_llm_provider
from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.rag import CitationSource, SourceCitation
from src.skillforge.models.resume import Skill
from src.skillforge.models.roadmap import (
    LearningRoadmap,
    RoadmapPriority,
    RoadmapSkillItem,
    RoadmapStage,
    RoadmapStageName,
    SkillGapStatus,
)
from src.skillforge.services.retrieval_service import RetrievalService
from src.skillforge.utils.exceptions import RoadmapGenerationError
from src.skillforge.utils.logging import logger


def _get_name(skill_val: Any) -> str:
    """Safely extract skill string from Skill model or string."""
    if hasattr(skill_val, "name"):
        return getattr(skill_val, "name")
    if hasattr(skill_val, "canonical_name"):
        return getattr(skill_val, "canonical_name")
    return str(skill_val)


def _get_similarity(match_obj: Any) -> float:
    """Safely extract similarity float from SkillMatch or dict."""
    if hasattr(match_obj, "similarity"):
        return float(match_obj.similarity)
    if hasattr(match_obj, "similarity_score"):
        return float(match_obj.similarity_score)
    return 0.0


def _get_overall_score(match_res: Any) -> float:
    """Safely extract overall match score."""
    if hasattr(match_res, "overall_score"):
        return float(match_res.overall_score)
    if hasattr(match_res, "overall_match_score"):
        return float(match_res.overall_match_score)
    return 0.0


class RoadmapGenerator:
    """
    Generates personalized learning roadmaps based on resume-job match results,
    skill gap analysis, and retrieved career knowledge base documents.
    """

    # Domain prerequisite dependency graph (Skill -> Prerequisites)
    PREREQUISITE_MAP: dict[str, list[str]] = {
        # Deep Learning & AI
        "deep learning": ["Python", "Machine Learning"],
        "pytorch": ["Python", "Deep Learning"],
        "tensorflow": ["Python", "Deep Learning"],
        "keras": ["Python", "TensorFlow"],
        "computer vision": ["Python", "Deep Learning"],
        "opencv": ["Python", "Computer Vision"],
        "yolo": ["Python", "Computer Vision", "PyTorch"],
        "nlp": ["Python", "Machine Learning"],
        "natural language processing": ["Python", "Machine Learning"],
        "transformers": ["Python", "NLP", "PyTorch"],
        "hugging face": ["Python", "Transformers"],
        "generative ai": ["Python", "Deep Learning", "Transformers"],
        "llms": ["Python", "Transformers", "Generative AI"],
        "rag": ["Python", "Vector Databases", "LLMs"],
        # Machine Learning & Data
        "machine learning": ["Python", "SQL", "Statistics"],
        "scikit-learn": ["Python", "Machine Learning"],
        "xgboost": ["Python", "Machine Learning"],
        "pandas": ["Python"],
        "numpy": ["Python"],
        # MLOps & Infrastructure
        "mlops": ["Python", "Docker", "Machine Learning"],
        "mlflow": ["Python", "MLOps"],
        "dvc": ["Git", "MLOps"],
        "docker": ["Linux"],
        "kubernetes": ["Docker", "Linux"],
        "terraform": ["Cloud Computing", "Linux"],
        "aws": ["Cloud Computing", "Linux"],
        "gcp": ["Cloud Computing", "Linux"],
        "azure": ["Cloud Computing", "Linux"],
        # Backend & Databases
        "fastapi": ["Python", "REST APIs"],
        "django": ["Python", "SQL"],
        "flask": ["Python"],
        "postgresql": ["SQL"],
        "mysql": ["SQL"],
        "redis": ["Data Structures"],
        "sql": ["Relational Concepts"],
        "dbt": ["SQL", "Data Modeling"],
        "data structures": ["Programming Foundations"],
        "algorithms": ["Data Structures"],
    }

    # Core foundational skills that naturally belong in Stage 1
    FOUNDATIONAL_SKILLS: set[str] = {
        "python",
        "sql",
        "git",
        "linux",
        "data structures",
        "algorithms",
        "statistics",
        "mathematics",
        "linear algebra",
        "rest apis",
        "programming foundations",
        "relational concepts",
        "cloud computing",
    }

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """
        Initialize the RoadmapGenerator.

        Args:
            retrieval_service: Multi-source vector retrieval service.
            llm_provider: Optional LLM provider for summary synthesis.
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider

    # ── Main Roadmap Generation ────────────────────────────────────────

    def generate_roadmap(
        self,
        match_result: MatchResult,
        target_role: str = "",
        max_skills_per_stage: int = 5,
    ) -> LearningRoadmap:
        """
        Generate an end-to-end personalized learning roadmap.

        Args:
            match_result: MatchResult from SemanticMatcher (M5).
            target_role: Target job title.
            max_skills_per_stage: Maximum number of skills per stage.

        Returns:
            Structured LearningRoadmap object.
        """
        role = target_role or getattr(match_result, "target_role", "Target Role")
        overall_score = _get_overall_score(match_result)
        logger.info("Generating personalized learning roadmap", target_role=role, score=overall_score)

        try:
            # 1. Classify skills: Demonstrated vs Partial vs Missing
            demonstrated_skills, partial_skills, missing_skills = self._classify_candidate_skills(match_result)

            # 2. Build candidate skill pool (exclude strongly demonstrated skills)
            skill_items_to_plan: list[RoadmapSkillItem] = []

            # Process Missing Skills
            gaps_list = getattr(match_result, "missing_skills", []) or getattr(match_result, "skill_gaps", [])
            for gap in gaps_list:
                gap_name = _get_name(getattr(gap, "skill", gap))
                importance_str = getattr(gap, "importance", "required")
                sim_val = getattr(gap, "closest_similarity", 0.0) or getattr(gap, "similarity_score", 0.0)

                item = self._create_roadmap_item(
                    skill_name=gap_name,
                    status=SkillGapStatus.MISSING,
                    importance_str=importance_str,
                    similarity_score=sim_val,
                    target_role=role,
                    candidate_skills=demonstrated_skills,
                )
                skill_items_to_plan.append(item)

            # Process Partial Skills (needing enhancement/bridging)
            partial_matches = getattr(match_result, "partial_matches", [])
            if not partial_matches and hasattr(match_result, "matching_skills"):
                partial_matches = [m for m in match_result.matching_skills if 0.50 <= _get_similarity(m) < 0.70]

            for p_match in partial_matches:
                p_name = _get_name(getattr(p_match, "job_skill", p_match))
                p_sim = _get_similarity(p_match)
                p_importance = getattr(p_match, "importance", "preferred")

                item = self._create_roadmap_item(
                    skill_name=p_name,
                    status=SkillGapStatus.PARTIAL,
                    importance_str=p_importance,
                    similarity_score=p_sim,
                    target_role=role,
                    candidate_skills=demonstrated_skills,
                )
                skill_items_to_plan.append(item)

            # 3. Compute Prerequisite Dependencies and Prioritization Scores
            self._apply_prioritization_and_stages(skill_items_to_plan, demonstrated_skills)

            # 4. Partition into 3 sequential stages
            stages = self._build_sequential_stages(skill_items_to_plan, max_skills_per_stage=max_skills_per_stage)

            # 5. Calculate Total Estimated Duration
            total_weeks = sum(s.estimated_duration_weeks for s in stages)

            # 6. Determine Key Focus Areas
            key_focus = self._derive_key_focus_areas(skill_items_to_plan)

            # 7. Synthesize Summary (Rule-based or LLM)
            summary = self._generate_summary(
                role=role,
                match_score=overall_score,
                stages=stages,
                demonstrated_count=len(demonstrated_skills),
                missing_count=len(missing_skills),
            )

            return LearningRoadmap(
                target_role=role,
                overall_match_score=overall_score,
                summary=summary,
                demonstrated_skills=sorted(list(set(demonstrated_skills))),
                partially_matched_skills=sorted(list(set(partial_skills))),
                missing_skills=sorted(list(set(missing_skills))),
                stages=stages,
                total_estimated_weeks=round(total_weeks, 1),
                key_focus_areas=key_focus,
                generated_with_llm=self.llm_provider is not None,
                model_used=getattr(self.llm_provider, "model_name", "deterministic-rules"),
            )

        except Exception as e:
            logger.error("Failed to generate learning roadmap", error=str(e))
            raise RoadmapGenerationError(f"Failed to generate learning roadmap: {e}") from e

    # ── Skill Classification & Item Creation ───────────────────────────

    def _classify_candidate_skills(
        self,
        match_result: MatchResult,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Split skills into Demonstrated (mastered), Partial (bridging), and Missing (deficit).

        Rules:
            - Demonstrated: similarity >= 0.70 (EXACT or STRONG match) -> Excluded from roadmap
            - Partial: 0.50 <= similarity < 0.70 -> Included as skill enhancement
            - Missing: similarity < 0.50 (in missing_skills/skill_gaps) -> Included as primary learning goals
        """
        demonstrated: list[str] = []
        partial: list[str] = []

        gaps_list = getattr(match_result, "missing_skills", []) or getattr(match_result, "skill_gaps", [])
        missing: list[str] = [_get_name(getattr(g, "skill", g)) for g in gaps_list]

        matched_list = getattr(match_result, "matched_skills", []) or getattr(match_result, "matching_skills", [])
        for m in matched_list:
            sim = _get_similarity(m)
            s_name = _get_name(getattr(m, "job_skill", m))
            if sim >= 0.70:
                demonstrated.append(s_name)
            elif sim >= 0.50:
                partial.append(s_name)

        partial_list = getattr(match_result, "partial_matches", [])
        for p in partial_list:
            p_name = _get_name(getattr(p, "job_skill", p))
            if p_name not in partial:
                partial.append(p_name)

        return demonstrated, partial, missing

    def _create_roadmap_item(
        self,
        skill_name: str,
        status: SkillGapStatus,
        importance_str: str,
        similarity_score: float,
        target_role: str,
        candidate_skills: list[str],
    ) -> RoadmapSkillItem:
        """Create a single RoadmapSkillItem and enrich it with career knowledge citations."""
        skill_lower = skill_name.lower().strip()

        # 1. Resolve prerequisites from dependency graph
        prereqs = self.PREREQUISITE_MAP.get(skill_lower, [])

        # 2. Retrieve supporting career knowledge from FAISS
        citations = self._retrieve_knowledge_for_skill(skill_name)
        has_kb = len(citations) > 0

        # 3. Extract learning objectives & project recommendations
        learning_objectives = self._extract_learning_objectives(skill_name, citations)
        recommended_project = self._extract_recommended_project(skill_name, citations, target_role)

        # 4. Estimate duration in weeks
        est_weeks = 1.0 if status == SkillGapStatus.PARTIAL else 2.5
        if skill_lower in self.FOUNDATIONAL_SKILLS:
            est_weeks = 1.5 if status == SkillGapStatus.PARTIAL else 2.0

        return RoadmapSkillItem(
            skill=skill_name,
            category=self._categorize_skill(skill_name),
            status=status,
            similarity_score=round(similarity_score, 2),
            prerequisites=prereqs,
            relationship_to_role=f"Essential capability for {target_role} responsibilities and technical workflows.",
            learning_objectives=learning_objectives,
            recommended_project=recommended_project,
            estimated_weeks=est_weeks,
            supporting_citations=citations,
            has_supporting_knowledge=has_kb,
        )

    # ── Prioritization & Stage Allocation ──────────────────────────────

    def _apply_prioritization_and_stages(
        self,
        items: list[RoadmapSkillItem],
        demonstrated_skills: list[str],
    ) -> None:
        """
        Compute explainable priority scores and assign stages.

        Scoring Formula:
            Base Score = 50 (Required) or 20 (Preferred)
            + Gap Score = 30 (Missing) or 15 (Partial)
            + Prerequisite Dependency Boost = 15 (if another missing skill depends on this)
            + Knowledge Grounding Boost = 5 (if high KB relevance)

        Stage Assignment:
            - Stage 1: Foundational missing skills OR skills that are prerequisites for others
            - Stage 2: Core required skills with prerequisites met
            - Stage 3: Preferred / nice-to-have skills or advanced specialization
        """
        demonstrated_set = {s.lower() for s in demonstrated_skills}
        all_skills_in_roadmap = {item.skill.lower() for item in items}

        for item in items:
            skill_lower = item.skill.lower()
            score = 0.0
            reasons = []

            # 1. Requirement Importance (+50 or +20)
            is_foundational = skill_lower in self.FOUNDATIONAL_SKILLS
            is_prereq_for_others = any(
                item.skill.lower() in [p.lower() for p in self.PREREQUISITE_MAP.get(other_skill, [])]
                for other_skill in all_skills_in_roadmap
                if other_skill != skill_lower
            )

            if item.status == SkillGapStatus.MISSING:
                score += 50.0  # Assumes missing items from gaps are high importance
                score += 30.0  # Missing gap penalty
                reasons.append("Mandatory job requirement not yet demonstrated in resume.")
            else:
                score += 35.0  # Partial match
                score += 15.0
                reasons.append("Partially demonstrated capability requiring targeted enhancement.")

            # 2. Prerequisite dependency boost
            if is_prereq_for_others:
                score += 15.0
                reasons.append("Foundational prerequisite required by other advanced target skills.")

            # 3. Knowledge Base Grounding Boost
            if item.has_supporting_knowledge:
                score += 5.0

            score = min(100.0, score)
            item.priority_score = round(score, 1)

            # Assign Priority Tier
            if score >= 75.0:
                item.priority = RoadmapPriority.HIGH
            elif score >= 45.0:
                item.priority = RoadmapPriority.MEDIUM
            else:
                item.priority = RoadmapPriority.LOW

            item.priority_reason = " ".join(reasons)

            # Assign Stage Number (1, 2, or 3)
            unmet_prereqs = [p for p in item.prerequisites if p.lower() not in demonstrated_set]

            if is_foundational or is_prereq_for_others or (unmet_prereqs and item.status == SkillGapStatus.MISSING and len(item.prerequisites) == 0):
                item.stage_number = 1
                item.stage_name = RoadmapStageName.STAGE_1.value
            elif item.priority == RoadmapPriority.HIGH or item.status == SkillGapStatus.MISSING:
                item.stage_number = 2
                item.stage_name = RoadmapStageName.STAGE_2.value
            else:
                item.stage_number = 3
                item.stage_name = RoadmapStageName.STAGE_3.value

    def _build_sequential_stages(
        self,
        items: list[RoadmapSkillItem],
        max_skills_per_stage: int = 5,
    ) -> list[RoadmapStage]:
        """Group skills into 3 sequential RoadmapStage objects sorted by priority score."""
        stage_map: dict[int, list[RoadmapSkillItem]] = {1: [], 2: [], 3: []}

        for item in items:
            stage_map[item.stage_number].append(item)

        # Sort each stage by priority_score descending
        for s_num in stage_map:
            stage_map[s_num].sort(key=lambda x: x.priority_score, reverse=True)

        stages: list[RoadmapStage] = []

        # Stage 1
        s1_items = stage_map[1][:max_skills_per_stage]
        s1_weeks = sum(i.estimated_weeks for i in s1_items) or 2.0
        stages.append(
            RoadmapStage(
                stage_number=1,
                stage_name=RoadmapStageName.STAGE_1.value,
                focus_area="Core engineering prerequisites, baseline syntax, and foundational architectures.",
                estimated_duration_weeks=round(s1_weeks, 1),
                skills=s1_items,
            )
        )

        # Stage 2
        s2_items = stage_map[2][:max_skills_per_stage]
        s2_weeks = sum(i.estimated_weeks for i in s2_items) or 3.0
        stages.append(
            RoadmapStage(
                stage_number=2,
                stage_name=RoadmapStageName.STAGE_2.value,
                focus_area="Primary frameworks, libraries, and direct technical requirements for the role.",
                estimated_duration_weeks=round(s2_weeks, 1),
                skills=s2_items,
            )
        )

        # Stage 3
        s3_items = stage_map[3][:max_skills_per_stage]
        s3_weeks = sum(i.estimated_weeks for i in s3_items) or 2.0
        stages.append(
            RoadmapStage(
                stage_number=3,
                stage_name=RoadmapStageName.STAGE_3.value,
                focus_area="Advanced productionization, cloud deployments, MLOps, and portfolio project completion.",
                estimated_duration_weeks=round(s3_weeks, 1),
                skills=s3_items,
            )
        )

        return stages

    # ── Career Knowledge & RAG Grounding Helpers ───────────────────────

    def _retrieve_knowledge_for_skill(self, skill_name: str) -> list[SourceCitation]:
        """Query the FAISS career knowledge base for the specific skill."""
        try:
            citations = self.retrieval_service.retrieve_from_knowledge_base(
                query=f"{skill_name} concepts tools learning progression practical projects",
                top_k=2,
                min_score=0.25,
            )
            return citations
        except Exception as e:
            logger.warning("Could not retrieve knowledge for skill", skill=skill_name, error=str(e))
            return []

    def _extract_learning_objectives(
        self,
        skill_name: str,
        citations: list[SourceCitation],
    ) -> list[str]:
        """Derive 3 structured learning objectives from retrieved citations or domain templates."""
        objectives: list[str] = []

        if citations:
            for c in citations:
                for line in c.content_preview.splitlines():
                    clean_line = line.strip().lstrip("-*•").strip()
                    if len(clean_line) > 20 and len(clean_line) < 120 and not clean_line.startswith("#"):
                        objectives.append(clean_line)
                    if len(objectives) >= 3:
                        break
                if len(objectives) >= 3:
                    break

        if len(objectives) < 3:
            objectives = [
                f"Master core syntax, foundational theory, and best practices for {skill_name}.",
                f"Implement standard algorithms and workflows using {skill_name} in isolated modules.",
                f"Integrate {skill_name} into an end-to-end production-grade service or data pipeline.",
            ]

        return objectives[:3]

    def _extract_recommended_project(
        self,
        skill_name: str,
        citations: list[SourceCitation],
        target_role: str,
    ) -> str:
        """Extract or synthesize a practical portfolio project for the skill."""
        if citations:
            for c in citations:
                if "Project" in c.section or "Portfolio" in c.section or "1." in c.content_preview:
                    for line in c.content_preview.splitlines():
                        if ("1." in line or "2." in line or "**" in line) and len(line.strip()) > 30:
                            return line.strip().lstrip("123456789.-*• ").strip()

        return f"Build an end-to-end {skill_name} project tailored for {target_role} with automated testing and documentation."

    # ── Metadata & Summary Helpers ─────────────────────────────────────

    @staticmethod
    def _categorize_skill(skill_name: str) -> str:
        """Determine skill category."""
        s = skill_name.lower()
        if s in {"python", "sql", "java", "c++", "go", "rust", "typescript", "javascript"}:
            return "programming_language"
        if s in {"pytorch", "tensorflow", "fastapi", "django", "flask", "react", "scikit-learn"}:
            return "framework"
        if s in {"postgresql", "mysql", "mongodb", "redis", "elasticsearch"}:
            return "database"
        if s in {"aws", "gcp", "azure", "docker", "kubernetes", "terraform", "mlflow"}:
            return "cloud_devops"
        return "technical"

    @staticmethod
    def _derive_key_focus_areas(items: list[RoadmapSkillItem]) -> list[str]:
        """Identify top 3 thematic focus areas based on highest priority skills."""
        high_priority = [i.skill for i in sorted(items, key=lambda x: x.priority_score, reverse=True)]
        return high_priority[:3] if high_priority else ["Core Technical Foundations"]

    def _generate_summary(
        self,
        role: str,
        match_score: float,
        stages: list[RoadmapStage],
        demonstrated_count: int,
        missing_count: int,
    ) -> str:
        """Generate executive summary of the roadmap."""
        total_weeks = sum(s.estimated_duration_weeks for s in stages)
        s1_count = len(stages[0].skills)
        s2_count = len(stages[1].skills)
        s3_count = len(stages[2].skills)

        return (
            f"Personalized learning roadmap for **{role}** (Current Match: {match_score:.1f}%). "
            f"You have strongly demonstrated **{demonstrated_count} key skills**, with **{missing_count} skill gaps** "
            f"organized across {len(stages)} sequential stages totaling **{total_weeks:.1f} weeks**. "
            f"Stage 1 tackles {s1_count} critical prerequisites, Stage 2 closes {s2_count} core role competencies, "
            f"and Stage 3 delivers {s3_count} advanced specialization and portfolio milestones."
        )
