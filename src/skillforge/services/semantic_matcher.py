"""
SkillForge AI — Semantic Matching & Explainable Scoring Service.

Computes a transparent, multi-factor match between a candidate's resume
and a target job description. The system goes beyond naive keyword counting
by combining:
    1. Semantic Skill Matching (embeddings + taxonomy aliases)
    2. Required vs. Preferred Skill Coverage
    3. Content Semantic Similarity (dense text embeddings)
    4. Relevant Experience & Seniority Alignment

Every score is accompanied by an explainable breakdown showing exact
weights, sub-scores, and reasoning.

Usage:
    from src.skillforge.services.semantic_matcher import SemanticMatcher

    matcher = SemanticMatcher()
    result = matcher.match_resume_to_job(
        resume=resume_analysis,
        job_description=jd_text,
        target_role="Senior Backend Engineer",
    )
    print(f"Overall Match: {result.overall_score:.1f}%")
    print(result.summary)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.resume import ResumeAnalysis, ResumeSection, Skill
from src.skillforge.services.skill_extractor import SkillExtractor
from src.skillforge.utils.exceptions import MatchingError
from src.skillforge.utils.logging import logger


class SemanticMatcher:
    """
    Computes a transparent, explainable match between resumes and job descriptions.

    Scoring Methodology:
        Overall Score = (w_skills * S_skills)
                      + (w_req * S_req)
                      + (w_semantic * S_semantic)
                      + (w_exp * S_exp)

        - S_skills (40%): Weighted average of skill similarity (exact, strong, partial).
        - S_req (25%): Percentage of mandatory/required job skills satisfied.
        - S_semantic (20%): Dense semantic similarity between resume and JD texts.
        - S_exp (15%): Experience years and seniority alignment.
    """

    # Similarity thresholds for classifying matches
    STRONG_MATCH_THRESHOLD = 0.70    # >= 0.70 is a strong match
    PARTIAL_MATCH_THRESHOLD = 0.50   # 0.50 - 0.69 is a partial match
    EXACT_MATCH_THRESHOLD = 0.95     # >= 0.95 or identical canonical name is exact

    # Multi-factor weights (must sum to 1.0)
    WEIGHT_SKILL_MATCH = 0.40
    WEIGHT_REQUIRED_COVERAGE = 0.25
    WEIGHT_SEMANTIC_SIMILARITY = 0.20
    WEIGHT_EXPERIENCE_ALIGNMENT = 0.15

    # Skill importance multipliers for skill match scoring
    IMPORTANCE_WEIGHTS = {
        "required": 1.5,
        "preferred": 1.0,
        "nice_to_have": 0.8,
    }

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        skill_extractor: SkillExtractor | None = None,
        strong_threshold: float | None = None,
        partial_threshold: float | None = None,
    ) -> None:
        """
        Initialize the semantic matcher.

        Args:
            embedding_engine: Engine for generating embeddings.
            skill_extractor: Skill extractor for parsing skills from raw text.
            strong_threshold: Override the strong match threshold.
            partial_threshold: Override the partial match threshold.
        """
        self.engine = embedding_engine or EmbeddingEngine()
        self.extractor = skill_extractor or SkillExtractor(use_nlp=True)

        if strong_threshold is not None:
            self.STRONG_MATCH_THRESHOLD = strong_threshold
        if partial_threshold is not None:
            self.PARTIAL_MATCH_THRESHOLD = partial_threshold

    # ── Top-Level Unified Matching API ─────────────────────────────────

    def match_resume_to_job(
        self,
        resume: ResumeAnalysis | str,
        job_description: str,
        target_role: str = "",
    ) -> MatchResult:
        """
        Perform an end-to-end multi-factor match between a resume and a job description.

        Args:
            resume: Either a processed ResumeAnalysis object or raw resume text.
            job_description: Full text of the target job description.
            target_role: Optional target role title (e.g. "Senior Backend Engineer").

        Returns:
            MatchResult with overall score, classified skills, experience breakdown,
            and explainable summary.
        """
        if not job_description or not job_description.strip():
            logger.warning("Empty job description provided for matching")
            return MatchResult(
                overall_score=0.0,
                summary="No job description provided for matching.",
            )

        # Extract resume text and sections
        if isinstance(resume, ResumeAnalysis):
            resume_text = resume.cleaned_text or resume.raw_text
            resume_sections = resume.sections
            resume_skills = (
                resume.skills
                if resume.skills
                else self.extractor.extract_skills(resume_text, source="resume")
            )
        else:
            resume_text = str(resume)
            resume_sections = []
            resume_skills = self.extractor.extract_skills(resume_text, source="resume")

        if not resume_text.strip():
            logger.warning("Empty resume provided for matching")
            job_skills_raw = self.extractor.extract_skills(
                job_description, source="job_description"
            )
            return MatchResult(
                overall_score=0.0,
                missing_skills=[
                    SkillGap(skill=js, importance="required") for js in job_skills_raw
                ],
                summary="No resume text provided. All job requirements are missing.",
            )

        # Extract job skills and classify requirement importance
        job_skills_raw = self.extractor.extract_skills(
            job_description, source="job_description"
        )
        job_skills_classified = self.classify_job_skill_importance(
            job_description, job_skills_raw
        )

        # Run multi-factor match
        return self._compute_comprehensive_match(
            resume_skills=resume_skills,
            job_skills_classified=job_skills_classified,
            resume_text=resume_text,
            job_description=job_description,
            resume_sections=resume_sections,
            target_role=target_role,
        )

    def match(
        self,
        resume_skills: list[Skill],
        job_skills: list[Skill],
        resume_text: str = "",
        job_description: str = "",
    ) -> MatchResult:
        """
        Match resume skills against job description skills.

        Backward-compatible method matching skill lists directly, with optional
        full-text context.

        Args:
            resume_skills: Extracted skills from the resume.
            job_skills: Extracted skills from the job description.
            resume_text: Optional full text of the resume.
            job_description: Optional full text of the job description.

        Returns:
            MatchResult with overall score and classified skills.
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
                missing_skills=[SkillGap(skill=js) for js in job_skills],
                summary="No resume skills found. All job requirements are gaps.",
            )

        # Classify skill importance if job description text is provided
        if job_description:
            job_skills_classified = self.classify_job_skill_importance(
                job_description, job_skills
            )
        else:
            job_skills_classified = [(s, "required") for s in job_skills]

        return self._compute_comprehensive_match(
            resume_skills=resume_skills,
            job_skills_classified=job_skills_classified,
            resume_text=resume_text,
            job_description=job_description,
            resume_sections=[],
            target_role="",
        )

    def match_texts(
        self,
        resume_text: str,
        job_text: str,
    ) -> float:
        """
        Compute a simple semantic cosine similarity between two texts.

        Args:
            resume_text: Resume text or excerpt.
            job_text: Job description text or excerpt.

        Returns:
            Cosine similarity score in [0, 1].
        """
        if not resume_text or not job_text:
            return 0.0
        return max(0.0, self.engine.similarity_between_texts(resume_text, job_text))

    # ── Core Comprehensive Matching Logic ──────────────────────────────

    def _compute_comprehensive_match(
        self,
        resume_skills: list[Skill],
        job_skills_classified: list[tuple[Skill, str]],
        resume_text: str,
        job_description: str,
        resume_sections: list[ResumeSection],
        target_role: str,
    ) -> MatchResult:
        """Internal computation of the multi-factor match."""
        job_skills = [s for s, _ in job_skills_classified]
        skill_importance_map = {s.name.lower(): imp for s, imp in job_skills_classified}

        # 1. Skill Embeddings & Pairwise Similarity Matrix
        resume_names = [s.name for s in resume_skills]
        job_names = [s.name for s in job_skills]

        resume_embeddings = self.engine.encode(resume_names)
        job_embeddings = self.engine.encode(job_names)

        sim_matrix = self.engine.pairwise_cosine_similarity(
            job_embeddings, resume_embeddings
        )

        # 2. Skill Classification (Strong, Partial, Gap)
        matched_skills: list[SkillMatch] = []
        partial_matches: list[SkillMatch] = []
        missing_skills: list[SkillGap] = []

        for j, job_skill in enumerate(job_skills):
            best_resume_idx = int(np.argmax(sim_matrix[j]))
            best_similarity = float(np.clip(sim_matrix[j, best_resume_idx], 0.0, 1.0))
            best_resume_skill = resume_skills[best_resume_idx]
            importance = skill_importance_map.get(job_skill.name.lower(), "required")

            # Check if canonical names match exactly
            is_exact = job_skill.name.lower() == best_resume_skill.name.lower()
            if is_exact:
                match_type = "exact"
                best_similarity = max(best_similarity, 1.0)
            elif best_similarity >= self.EXACT_MATCH_THRESHOLD:
                match_type = "alias"
            elif best_similarity >= self.STRONG_MATCH_THRESHOLD:
                match_type = "strong_semantic"
            else:
                match_type = "partial_semantic"

            if is_exact or best_similarity >= self.STRONG_MATCH_THRESHOLD:
                matched_skills.append(
                    SkillMatch(
                        resume_skill=best_resume_skill,
                        job_skill=job_skill,
                        similarity=best_similarity,
                        match_type=match_type,
                    )
                )
            elif best_similarity >= self.PARTIAL_MATCH_THRESHOLD:
                partial_matches.append(
                    SkillMatch(
                        resume_skill=best_resume_skill,
                        job_skill=job_skill,
                        similarity=best_similarity,
                        match_type=match_type,
                    )
                )
            else:
                recommendation = self._generate_gap_recommendation(
                    job_skill=job_skill,
                    closest_skill=best_resume_skill.name,
                    similarity=best_similarity,
                )
                missing_skills.append(
                    SkillGap(
                        skill=job_skill,
                        importance=importance,
                        closest_resume_skill=best_resume_skill.name,
                        closest_similarity=best_similarity,
                        recommendation=recommendation,
                    )
                )

        # 3. Factor 1: Skill Match Score (0 - 100)
        skill_match_score = self._compute_skill_match_score(
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            importance_map=skill_importance_map,
        )

        # 4. Factor 2: Required Skills Coverage (0 - 100)
        required_coverage_score, req_total, req_satisfied = self._compute_required_coverage(
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            importance_map=skill_importance_map,
        )

        # 5. Factor 3: Content Semantic Similarity Score (0 - 100)
        semantic_similarity_score, raw_semantic_sim = self._compute_semantic_similarity_score(
            resume_text=resume_text,
            job_description=job_description,
            matched_skills=matched_skills,
        )

        # 6. Factor 4: Experience & Seniority Alignment Score (0 - 100)
        experience_analysis = self._analyze_experience(
            resume_text=resume_text,
            job_description=job_description,
            resume_sections=resume_sections,
            target_role=target_role,
        )
        experience_score = experience_analysis["experience_score"]

        # 7. Overall Weighted Explainable Score
        # If full text was not provided (standalone skill matching), scale skill & coverage weights
        if not resume_text or not job_description:
            overall_score = 0.70 * skill_match_score + 0.30 * required_coverage_score
            weights_used = {
                "skill_match": 0.70,
                "required_coverage": 0.30,
                "semantic_similarity": 0.0,
                "experience_alignment": 0.0,
            }
        else:
            overall_score = (
                self.WEIGHT_SKILL_MATCH * skill_match_score
                + self.WEIGHT_REQUIRED_COVERAGE * required_coverage_score
                + self.WEIGHT_SEMANTIC_SIMILARITY * semantic_similarity_score
                + self.WEIGHT_EXPERIENCE_ALIGNMENT * experience_score
            )
            weights_used = {
                "skill_match": self.WEIGHT_SKILL_MATCH,
                "required_coverage": self.WEIGHT_REQUIRED_COVERAGE,
                "semantic_similarity": self.WEIGHT_SEMANTIC_SIMILARITY,
                "experience_alignment": self.WEIGHT_EXPERIENCE_ALIGNMENT,
            }

        overall_score = float(np.clip(overall_score, 0.0, 100.0))

        # 8. Construct Detailed Breakdown
        score_breakdown: dict[str, Any] = {
            "overall": round(overall_score, 1),
            "skill_match_score": round(skill_match_score, 1),
            "required_skills_coverage": round(required_coverage_score, 1),
            "semantic_similarity_score": round(semantic_similarity_score, 1),
            "raw_semantic_similarity": round(raw_semantic_sim, 3),
            "experience_alignment_score": round(experience_score, 1),
            "weights": weights_used,
            "matched_count": len(matched_skills),
            "partial_count": len(partial_matches),
            "missing_count": len(missing_skills),
            "total_job_skills": len(job_skills),
            "required_skills_total": req_total,
            "required_skills_satisfied": req_satisfied,
            # Backward-compatibility keys for M4 tests
            "strong_component": round(skill_match_score, 1),
            "partial_component": round(
                (len(partial_matches) / max(1, len(job_skills))) * 100, 1
            ),
            "coverage": round(required_coverage_score, 1),
        }

        # 9. Build Comprehensive Human-Readable Summary
        summary = self._build_explainable_summary(
            overall_score=overall_score,
            score_breakdown=score_breakdown,
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            experience_analysis=experience_analysis,
        )

        return MatchResult(
            overall_score=round(overall_score, 1),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            partial_matches=partial_matches,
            score_breakdown=score_breakdown,
            summary=summary,
            experience_analysis=experience_analysis,
            required_skills_coverage=round(required_coverage_score, 1),
            semantic_similarity_score=round(semantic_similarity_score, 1),
        )

    # ── Component Scoring Formulas ─────────────────────────────────────

    def _compute_skill_match_score(
        self,
        matched_skills: list[SkillMatch],
        partial_matches: list[SkillMatch],
        missing_skills: list[SkillGap],
        importance_map: dict[str, str],
    ) -> float:
        """
        Compute the Skill Match Quality Score (0–100%).

        Each job skill receives a score:
            - Strong match: similarity (0.70 to 1.0)
            - Partial match: similarity * 0.65 (partial credit)
            - Missing gap: 0.0

        Each skill score is weighted by its importance multiplier.
        """
        total_weighted_points = 0.0
        total_max_weights = 0.0

        for match in matched_skills:
            imp = importance_map.get(match.job_skill.name.lower(), "required")
            w = self.IMPORTANCE_WEIGHTS.get(imp, 1.0)
            total_weighted_points += match.similarity * w
            total_max_weights += w

        for match in partial_matches:
            imp = importance_map.get(match.job_skill.name.lower(), "required")
            w = self.IMPORTANCE_WEIGHTS.get(imp, 1.0)
            # Partial credit scaled by similarity
            total_weighted_points += (match.similarity * 0.65) * w
            total_max_weights += w

        for gap in missing_skills:
            imp = importance_map.get(gap.skill.name.lower(), "required")
            w = self.IMPORTANCE_WEIGHTS.get(imp, 1.0)
            total_max_weights += w

        if total_max_weights == 0:
            return 0.0

        return (total_weighted_points / total_max_weights) * 100.0

    def _compute_required_coverage(
        self,
        matched_skills: list[SkillMatch],
        partial_matches: list[SkillMatch],
        missing_skills: list[SkillGap],
        importance_map: dict[str, str],
    ) -> tuple[float, int, float]:
        """
        Compute Required Skill Coverage (0–100%).

        Returns:
            Tuple of (coverage_percentage, total_required_count, satisfied_required_count).
        """
        req_matched = [
            m for m in matched_skills
            if importance_map.get(m.job_skill.name.lower(), "required") == "required"
        ]
        req_partial = [
            m for m in partial_matches
            if importance_map.get(m.job_skill.name.lower(), "required") == "required"
        ]
        req_missing = [
            g for g in missing_skills
            if importance_map.get(g.skill.name.lower(), "required") == "required"
        ]

        total_required = len(req_matched) + len(req_partial) + len(req_missing)

        if total_required == 0:
            # If no skills explicitly flagged as required, compute over all skills
            total_all = len(matched_skills) + len(partial_matches) + len(missing_skills)
            if total_all == 0:
                return 0.0, 0, 0.0
            satisfied = len(matched_skills) + 0.5 * len(partial_matches)
            return (satisfied / total_all) * 100.0, total_all, satisfied

        satisfied_count = len(req_matched) + (0.5 * len(req_partial))
        coverage_score = (satisfied_count / total_required) * 100.0
        return coverage_score, total_required, satisfied_count

    def _compute_semantic_similarity_score(
        self,
        resume_text: str,
        job_description: str,
        matched_skills: list[SkillMatch],
    ) -> tuple[float, float]:
        """
        Compute dense semantic similarity score (0–100%) between resume and JD.

        Normalizes raw cosine similarity into a standardized 0–100 scale.
        """
        if not resume_text or not job_description:
            # Fall back to average matched similarity if full text unavailable
            if matched_skills:
                avg_sim = float(np.mean([m.similarity for m in matched_skills]))
                return avg_sim * 100.0, avg_sim
            return 0.0, 0.0

        raw_sim = float(
            np.clip(self.engine.similarity_between_texts(resume_text, job_description), 0.0, 1.0)
        )

        # Min-max normalization for sentence transformer cosine similarity
        # (typical relevant pairs score 0.25–0.85)
        normalized = np.clip((raw_sim - 0.20) / (0.80 - 0.20), 0.0, 1.0) * 100.0
        return float(normalized), raw_sim

    # ── Experience & Requirement Analysis ──────────────────────────────

    def _analyze_experience(
        self,
        resume_text: str,
        job_description: str,
        resume_sections: list[ResumeSection],
        target_role: str,
    ) -> dict[str, Any]:
        """
        Analyze years of experience and seniority alignment.

        Returns:
            Dictionary containing detected years, required years, seniority levels,
            and an overall experience score (0–100).
        """
        req_years = self.extract_required_years(job_description)
        cand_years = self.extract_candidate_years(resume_text, resume_sections)

        job_seniority = self.extract_seniority_level(f"{target_role} {job_description}")
        cand_seniority = self.extract_seniority_level(resume_text)

        # Compute Experience Score
        score = 80.0  # Baseline

        if req_years is not None and cand_years is not None:
            if cand_years >= req_years:
                score = 100.0
            elif cand_years > 0:
                score = min(100.0, (cand_years / req_years) * 100.0)
            else:
                score = 50.0
        elif req_years is not None and cand_years is None:
            # Required years stated, but candidate years not explicitly detected
            score = 70.0
        elif cand_years is not None:
            # Candidate has experience, no specific JD years required
            score = 90.0 if cand_years >= 3 else 80.0

        # Seniority Alignment Adjustment
        seniority_hierarchy = {
            "entry": 1,
            "junior": 1,
            "mid": 2,
            "senior": 3,
            "lead": 4,
            "principal": 5,
            "staff": 5,
            "director": 6,
        }

        job_lvl = seniority_hierarchy.get(job_seniority, 0)
        cand_lvl = seniority_hierarchy.get(cand_seniority, 0)

        notes = []
        if req_years:
            notes.append(f"Job requires {req_years:g}+ years of experience.")
        if cand_years:
            notes.append(f"Candidate has approximately {cand_years:g} years of detected experience.")

        if job_lvl > 0 and cand_lvl > 0:
            if cand_lvl >= job_lvl:
                notes.append(f"Seniority aligned ({cand_seniority.title()} candidate for {job_seniority.title()} role).")
                score = min(100.0, score + 5.0)
            else:
                notes.append(f"Seniority gap: role targets {job_seniority.title()}, resume indicates {cand_seniority.title()}.")
                score = max(40.0, score - 15.0)

        return {
            "experience_score": round(score, 1),
            "years_required": req_years,
            "years_detected": cand_years,
            "job_seniority": job_seniority or "Not specified",
            "candidate_seniority": cand_seniority or "Not specified",
            "notes": notes,
        }

    # ── Extraction & Helper Utilities ──────────────────────────────────

    def extract_required_years(self, text: str) -> float | None:
        """
        Extract required years of experience from job description text.

        Examples: '5+ years of experience', '3-5 years', 'minimum 4 yrs'.
        """
        if not text:
            return None

        # Pattern 1: Range like "3-5 years" or "3 to 5 years" -> take minimum
        range_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            text,
            re.IGNORECASE,
        )
        if range_match:
            return float(range_match.group(1))

        # Pattern 2: "5+ years of experience", "minimum 4 years"
        single_match = re.search(
            r"(?:minimum|at least|min\.?)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|hands-on|industry|relevant|software)",
            text,
            re.IGNORECASE,
        )
        if single_match:
            return float(single_match.group(1))

        # Pattern 3: "5+ years in backend"
        loose_match = re.search(
            r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:in|with|using|building)",
            text,
            re.IGNORECASE,
        )
        if loose_match:
            return float(loose_match.group(1))

        return None

    def extract_candidate_years(
        self,
        resume_text: str,
        sections: list[ResumeSection] | None = None,
    ) -> float | None:
        """
        Extract or estimate total years of experience from resume text.

        Combines explicit statements and experience date ranges.
        """
        if not resume_text:
            return None

        # 1. Check for explicit mention: "5+ years of experience"
        explicit = re.search(
            r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|in software|as a|professional)",
            resume_text,
            re.IGNORECASE,
        )
        if explicit:
            return float(explicit.group(1))

        # 2. Check year ranges in experience sections (e.g. 2018–2021, 2021-Present)
        current_year = datetime.now().year
        year_ranges = re.findall(
            r"\b(20\d{2}|19\d{2})\s*(?:–|-|to)\s*(20\d{2}|19\d{2}|present|current)\b",
            resume_text,
            re.IGNORECASE,
        )

        if year_ranges:
            total_span_years = 0.0
            for start_str, end_str in year_ranges:
                start_yr = int(start_str)
                end_yr = current_year if end_str.lower() in ("present", "current") else int(end_str)
                if end_yr >= start_yr:
                    total_span_years += max(1.0, float(end_yr - start_yr))

            if total_span_years > 0:
                # Cap estimated span at reasonable bounds
                return min(30.0, total_span_years)

        return None

    def extract_seniority_level(self, text: str) -> str:
        """Identify seniority level in text (e.g. 'Senior', 'Lead', 'Junior')."""
        text_lower = text.lower()

        if re.search(r"\b(principal|staff|architect)\b", text_lower):
            return "principal"
        if re.search(r"\b(lead|tech lead|team lead|engineering manager)\b", text_lower):
            return "lead"
        if re.search(r"\b(senior|sr\.?|lead developer)\b", text_lower):
            return "senior"
        if re.search(r"\b(junior|jr\.?|entry level|associate|intern|graduate)\b", text_lower):
            return "junior"
        if re.search(r"\b(mid-level|intermediate|mid level)\b", text_lower):
            return "mid"

        return ""

    def classify_job_skill_importance(
        self,
        job_description: str,
        skills: list[Skill],
    ) -> list[tuple[Skill, str]]:
        """
        Classify whether each extracted job skill is 'required' or 'nice_to_have'.

        Splits the job description into sections and checks for preferred/nice-to-have headers.
        """
        if not job_description or not skills:
            return [(s, "required") for s in skills]

        lines = job_description.split("\n")
        current_importance = "required"
        line_importance = []

        for line in lines:
            line_lower = line.strip().lower()
            if re.search(
                r"\b(nice to have|preferred|bonus|plus|good to have|desired|optional|additional skills)\b",
                line_lower,
            ):
                current_importance = "nice_to_have"
            elif re.search(
                r"\b(requirements?|required|qualifications?|must have|responsibilities|what you need|minimum qualifications)\b",
                line_lower,
            ):
                current_importance = "required"

            line_importance.append((line, current_importance))

        classified = []
        for skill in skills:
            names_to_check = {skill.name.lower()}
            defn = self.extractor.taxonomy.lookup(skill.name)
            if defn:
                names_to_check.add(defn.name.lower())
                names_to_check.update(a.lower() for a in defn.aliases)

            patterns = [r"\b" + re.escape(n) + r"\b" for n in names_to_check]
            combined_pattern = "|".join(patterns)

            assigned_importance = "required"
            for line, importance in line_importance:
                if re.search(combined_pattern, line.lower()):
                    assigned_importance = importance
                    break

            classified.append((skill, assigned_importance))

        return classified

    def _generate_gap_recommendation(
        self,
        job_skill: Skill,
        closest_skill: str,
        similarity: float,
    ) -> str:
        """Generate an actionable learning recommendation for a missing skill."""
        if similarity >= self.PARTIAL_MATCH_THRESHOLD:
            return (
                f"You have related experience with '{closest_skill}' ({similarity:.0%} similarity). "
                f"Bridge this gap by highlighting transferable concepts or completing a targeted tutorial."
            )
        return (
            f"Add projects, coursework, or practical examples demonstrating '{job_skill.name}' to your resume."
        )

    def _build_explainable_summary(
        self,
        overall_score: float,
        score_breakdown: dict[str, Any],
        matched_skills: list[SkillMatch],
        partial_matches: list[SkillMatch],
        missing_skills: list[SkillGap],
        experience_analysis: dict[str, Any],
    ) -> str:
        """Generate a transparent, explainable summary report."""
        if overall_score >= 80:
            tier = "🌟 Strong Match"
        elif overall_score >= 65:
            tier = "✅ Good Match"
        elif overall_score >= 50:
            tier = "⚡ Moderate Match"
        else:
            tier = "⚠️ Significant Gap"

        lines = [
            f"Match Score: {overall_score:.1f}% ({tier})",
            "",
            "📊 Score Breakdown:",
            f"  • Skill Match Quality: {score_breakdown['skill_match_score']:.1f}% (Weight: 40%)",
            f"  • Required Skills Coverage: {score_breakdown['required_skills_coverage']:.1f}% (Weight: 25%)",
            f"  • Content Semantic Similarity: {score_breakdown['semantic_similarity_score']:.1f}% (Weight: 20%)",
            f"  • Experience & Seniority Alignment: {score_breakdown['experience_alignment_score']:.1f}% (Weight: 15%)",
            f"🎯 Skills Summary: {len(matched_skills)} strong matches, {len(partial_matches)} partial matches, {len(missing_skills)} gaps.",
            f"  Strong matches: {len(matched_skills)}/{len(matched_skills) + len(partial_matches) + len(missing_skills)} job skills",
            f"  Partial matches: {len(partial_matches)}/{len(matched_skills) + len(partial_matches) + len(missing_skills)} job skills",
            f"  Gaps: {len(missing_skills)}/{len(matched_skills) + len(partial_matches) + len(missing_skills)} job skills",
        ]

        if matched_skills:
            lines.append("\nTop Matching Skills:")
            for m in sorted(matched_skills, key=lambda x: -x.similarity)[:5]:
                match_label = f"[{m.match_type.replace('_', ' ').title()}]"
                lines.append(
                    f"  ✅ {m.job_skill.name} ← {m.resume_skill.name} ({m.similarity:.0%}) {match_label}"
                )

        if partial_matches:
            lines.append("\nPartial Skill Matches:")
            for p in sorted(partial_matches, key=lambda x: -x.similarity)[:3]:
                lines.append(
                    f"  ⚡ {p.job_skill.name} ≈ {p.resume_skill.name} ({p.similarity:.0%})"
                )

        if missing_skills:
            lines.append("\nKey Skill Gaps:")
            for g in missing_skills[:5]:
                imp_badge = f"[{g.importance.replace('_', ' ').upper()}]"
                lines.append(f"  ❌ {g.skill.name} {imp_badge}: {g.recommendation}")

        if experience_analysis.get("notes"):
            lines.append("\nExperience Assessment:")
            for note in experience_analysis["notes"]:
                lines.append(f"  📌 {note}")

        return "\n".join(lines)
