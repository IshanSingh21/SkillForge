"""
SkillForge AI — Semantic Matching Service.

Uses sentence-transformer embeddings to compute meaningful similarity
between resume skills and job-description requirements. This goes far
beyond keyword matching — it captures that "team leadership" ≈ "people
management" and "React.js" ≈ "React framework".

Pipeline:
    1. Encode all resume skills and job skills as embeddings.
    2. Compute the pairwise cosine similarity matrix (M×N).
    3. For each job skill, find the best-matching resume skill.
    4. Classify matches: strong match, partial match, or gap.
    5. Compute a weighted overall match score.

Usage:
    from src.skillforge.services.semantic_matcher import SemanticMatcher

    matcher = SemanticMatcher()
    result = matcher.match(resume_skills, job_skills)
    print(f"Score: {result.overall_score:.1f}%")
"""

from __future__ import annotations

import numpy as np

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.resume import Skill
from src.skillforge.utils.exceptions import MatchingError
from src.skillforge.utils.logging import logger


class SemanticMatcher:
    """
    Computes semantic similarity between resume and job-description skills.

    Unlike keyword matching, semantic matching understands that:
        - "Python programming" ≈ "Python development"  (~0.92)
        - "Team leadership" ≈ "People management"      (~0.75)
        - "AWS" ≈ "Amazon Web Services"                 (~0.85)
        - "Machine Learning" ≈ "ML"                     (~0.80)

    These relationships are impossible to capture with string matching
    but are naturally encoded in sentence-transformer embeddings.
    """

    # Thresholds for classifying matches
    STRONG_MATCH_THRESHOLD = 0.70    # Above this = strong match
    PARTIAL_MATCH_THRESHOLD = 0.50   # Between partial and strong = partial match
    # Below partial = gap (missing skill)

    # Score weights
    WEIGHT_STRONG = 0.65    # Weight for strong match component
    WEIGHT_PARTIAL = 0.25   # Weight for partial match component
    WEIGHT_COVERAGE = 0.10  # Weight for overall coverage breadth

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        strong_threshold: float | None = None,
        partial_threshold: float | None = None,
    ) -> None:
        """
        Initialize the semantic matcher.

        Args:
            embedding_engine: Engine for generating embeddings. Created with
                              defaults if not provided.
            strong_threshold: Override the strong match threshold.
            partial_threshold: Override the partial match threshold.
        """
        self.engine = embedding_engine or EmbeddingEngine()

        if strong_threshold is not None:
            self.STRONG_MATCH_THRESHOLD = strong_threshold
        if partial_threshold is not None:
            self.PARTIAL_MATCH_THRESHOLD = partial_threshold

    def match(
        self,
        resume_skills: list[Skill],
        job_skills: list[Skill],
    ) -> MatchResult:
        """
        Compute the semantic match between resume skills and job skills.

        Args:
            resume_skills: Skills extracted from the resume.
            job_skills: Skills extracted from the job description.

        Returns:
            MatchResult with overall score, matched/missing/partial skills.

        Raises:
            MatchingError: If matching fails.
        """
        if not job_skills:
            logger.warning("No job skills to match against")
            return MatchResult(
                overall_score=0.0,
                summary="No job skills provided for matching.",
            )

        if not resume_skills:
            logger.warning("No resume skills to match")
            return MatchResult(
                overall_score=0.0,
                missing_skills=[
                    SkillGap(skill=js) for js in job_skills
                ],
                summary="No resume skills found. All job requirements are gaps.",
            )

        try:
            return self._compute_match(resume_skills, job_skills)
        except MatchingError:
            raise
        except Exception as e:
            logger.error("Semantic matching failed", error=str(e))
            raise MatchingError(
                f"Semantic matching failed: {e}",
                detail="An error occurred while computing skill similarity.",
            ) from e

    def match_texts(
        self,
        resume_text: str,
        job_text: str,
    ) -> float:
        """
        Compute a simple semantic similarity between two texts.

        This is a convenience method for quick comparison without
        going through the full skill extraction pipeline.

        Args:
            resume_text: Resume text or excerpt.
            job_text: Job description text or excerpt.

        Returns:
            Cosine similarity score in [0, 1].
        """
        return max(0.0, self.engine.similarity_between_texts(resume_text, job_text))

    # ── Core Matching Logic ────────────────────────────────────────────

    def _compute_match(
        self,
        resume_skills: list[Skill],
        job_skills: list[Skill],
    ) -> MatchResult:
        """Compute the full match result."""
        # Encode skill names as embeddings
        resume_names = [s.name for s in resume_skills]
        job_names = [s.name for s in job_skills]

        logger.info(
            "Computing semantic match",
            resume_skills=len(resume_names),
            job_skills=len(job_names),
        )

        resume_embeddings = self.engine.encode(resume_names)
        job_embeddings = self.engine.encode(job_names)

        # Compute pairwise similarity matrix: (num_job × num_resume)
        sim_matrix = self.engine.pairwise_cosine_similarity(
            job_embeddings, resume_embeddings
        )

        # Classify each job skill
        matched_skills: list[SkillMatch] = []
        partial_matches: list[SkillMatch] = []
        missing_skills: list[SkillGap] = []

        for j, job_skill in enumerate(job_skills):
            # Find the best-matching resume skill for this job skill
            best_resume_idx = int(np.argmax(sim_matrix[j]))
            best_similarity = float(np.clip(sim_matrix[j, best_resume_idx], 0.0, 1.0))

            best_resume_skill = resume_skills[best_resume_idx]

            if best_similarity >= self.STRONG_MATCH_THRESHOLD:
                matched_skills.append(
                    SkillMatch(
                        resume_skill=best_resume_skill,
                        job_skill=job_skill,
                        similarity=best_similarity,
                    )
                )
            elif best_similarity >= self.PARTIAL_MATCH_THRESHOLD:
                partial_matches.append(
                    SkillMatch(
                        resume_skill=best_resume_skill,
                        job_skill=job_skill,
                        similarity=best_similarity,
                    )
                )
            else:
                missing_skills.append(
                    SkillGap(
                        skill=job_skill,
                        closest_resume_skill=best_resume_skill.name,
                        closest_similarity=best_similarity,
                    )
                )

        # Compute weighted score
        total_job = len(job_skills)
        score_breakdown = self._compute_score(
            matched=len(matched_skills),
            partial=len(partial_matches),
            missing=len(missing_skills),
            total=total_job,
            matched_sims=[m.similarity for m in matched_skills],
            partial_sims=[m.similarity for m in partial_matches],
        )

        overall_score = score_breakdown["overall"]

        # Build summary
        summary = self._build_summary(
            overall_score, matched_skills, partial_matches, missing_skills
        )

        logger.info(
            "Semantic matching complete",
            score=f"{overall_score:.1f}",
            matched=len(matched_skills),
            partial=len(partial_matches),
            missing=len(missing_skills),
        )

        return MatchResult(
            overall_score=overall_score,
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            score_breakdown=score_breakdown,
            summary=summary,
        )

    def _compute_score(
        self,
        matched: int,
        partial: int,
        missing: int,
        total: int,
        matched_sims: list[float],
        partial_sims: list[float],
    ) -> dict[str, float]:
        """
        Compute the weighted overall match score.

        Formula:
            strong_component  = (num_strong / total) × avg_strong_similarity × 100
            partial_component = (num_partial / total) × avg_partial_similarity × 100
            coverage          = ((strong + partial) / total) × 100

            overall = strong_weight × strong_component
                    + partial_weight × partial_component
                    + coverage_weight × coverage
        """
        if total == 0:
            return {"overall": 0.0, "strong_component": 0.0,
                    "partial_component": 0.0, "coverage": 0.0}

        avg_strong_sim = float(np.mean(matched_sims)) if matched_sims else 0.0
        avg_partial_sim = float(np.mean(partial_sims)) if partial_sims else 0.0

        strong_component = (matched / total) * avg_strong_sim * 100
        partial_component = (partial / total) * avg_partial_sim * 100
        coverage = ((matched + partial) / total) * 100

        overall = (
            self.WEIGHT_STRONG * strong_component
            + self.WEIGHT_PARTIAL * partial_component
            + self.WEIGHT_COVERAGE * coverage
        )

        # Clamp to [0, 100]
        overall = max(0.0, min(100.0, overall))

        return {
            "overall": round(overall, 1),
            "strong_component": round(strong_component, 1),
            "partial_component": round(partial_component, 1),
            "coverage": round(coverage, 1),
            "avg_strong_similarity": round(avg_strong_sim, 3),
            "avg_partial_similarity": round(avg_partial_sim, 3),
            "matched_count": matched,
            "partial_count": partial,
            "missing_count": missing,
            "total_job_skills": total,
        }

    @staticmethod
    def _build_summary(
        score: float,
        matched: list[SkillMatch],
        partial: list[SkillMatch],
        missing: list[SkillGap],
    ) -> str:
        """Build a human-readable match summary."""
        total = len(matched) + len(partial) + len(missing)

        lines = [f"Match Score: {score:.1f}%"]
        lines.append(f"  Strong matches: {len(matched)}/{total} job skills")
        lines.append(f"  Partial matches: {len(partial)}/{total} job skills")
        lines.append(f"  Gaps: {len(missing)}/{total} job skills")

        if matched:
            lines.append("\nTop matches:")
            for m in sorted(matched, key=lambda x: -x.similarity)[:5]:
                lines.append(
                    f"  ✅ {m.job_skill.name} ← {m.resume_skill.name} "
                    f"({m.similarity:.0%})"
                )

        if missing:
            lines.append("\nKey gaps:")
            for g in missing[:5]:
                lines.append(f"  ❌ {g.skill.name} (not found in resume)")

        return "\n".join(lines)
