"""
SkillForge AI — Hybrid Skill Extraction Service.

Extracts skills from text using a three-layer hybrid approach:

    Layer 1: Taxonomy Pattern Matching (most reliable)
        Scans text for known skills from the curated taxonomy using
        multi-word phrase matching, then single-word matching with
        word-boundary awareness for short terms like "C", "R", "Go".

    Layer 2: spaCy NLP Analysis (when available)
        Uses spaCy's noun phrase extraction and named entity recognition
        to discover skill candidates not in the taxonomy. Candidates are
        then fuzzy-matched against the taxonomy to catch typos and variants.

    Layer 3: N-gram Heuristic Extraction (fallback)
        Extracts n-grams from text and matches them against the taxonomy
        using string similarity. Used when spaCy is not available.

The system is designed so Layer 1 alone provides solid results, with
Layers 2-3 adding incremental coverage. This makes the extraction
reliable without depending on any external API.

Usage:
    from src.skillforge.services.skill_extractor import SkillExtractor

    extractor = SkillExtractor()
    skills = extractor.extract_skills(resume_text, source="resume")
    for skill in skills:
        print(f"{skill.name} ({skill.category.value}) — {skill.confidence:.0%}")
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.skillforge.data.skill_taxonomy import SkillDefinition, SkillTaxonomy
from src.skillforge.models.resume import Skill, SkillCategory
from src.skillforge.utils.exceptions import SkillExtractionError
from src.skillforge.utils.logging import logger


class SkillExtractor:
    """
    Hybrid skill extractor combining taxonomy matching, NLP analysis,
    and fuzzy string matching.

    The extractor is deliberately LLM-free — it uses curated data and
    NLP techniques for deterministic, reproducible results. An LLM
    refinement layer can be added on top in a future milestone.
    """

    # Minimum fuzzy match ratio to consider a candidate a skill
    FUZZY_THRESHOLD = 0.85

    # Minimum confidence to include a skill in results
    MIN_CONFIDENCE = 0.5

    def __init__(
        self,
        taxonomy: SkillTaxonomy | None = None,
        use_nlp: bool = True,
    ) -> None:
        """
        Initialize the skill extractor.

        Args:
            taxonomy: Skill taxonomy to match against. Defaults to built-in.
            use_nlp: Whether to attempt loading spaCy for NLP analysis.
                     Gracefully falls back to pattern-only if unavailable.
        """
        self.taxonomy = taxonomy or SkillTaxonomy()
        self._nlp = None

        if use_nlp:
            self._nlp = self._load_spacy()

        # Pre-build search patterns from taxonomy
        self._multi_word_patterns = self._build_multi_word_patterns()
        self._single_word_patterns = self._build_single_word_patterns()

        logger.info(
            "SkillExtractor initialized",
            taxonomy_size=self.taxonomy.size,
            nlp_available=self._nlp is not None,
        )

    def extract_skills(
        self,
        text: str,
        source: str = "resume",
    ) -> list[Skill]:
        """
        Extract skills from text using the hybrid pipeline.

        Args:
            text: The text to extract skills from (resume or job description).
            source: Label for where this text came from ('resume', 'job_description').

        Returns:
            Deduplicated, sorted list of Skill objects.

        Raises:
            SkillExtractionError: If extraction completely fails.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for skill extraction")
            return []

        try:
            found_skills: dict[str, Skill] = {}  # canonical_name_lower → Skill

            # ── Layer 1: Taxonomy Pattern Matching ─────────────────
            pattern_skills = self._extract_by_pattern(text)
            for skill in pattern_skills:
                self._merge_skill(found_skills, skill)

            logger.debug(
                "Pattern matching complete",
                found=len(found_skills),
            )

            # ── Layer 2: spaCy NLP Analysis ────────────────────────
            if self._nlp is not None:
                nlp_skills = self._extract_by_nlp(text)
                for skill in nlp_skills:
                    self._merge_skill(found_skills, skill)

                logger.debug(
                    "NLP extraction complete",
                    additional=len(nlp_skills),
                )

            # ── Finalize ───────────────────────────────────────────
            result = list(found_skills.values())

            # Set source on all skills
            for skill in result:
                skill.source = source

            # Sort: high confidence first, then alphabetical
            result.sort(key=lambda s: (-s.confidence, s.name.lower()))

            logger.info(
                "Skill extraction complete",
                source=source,
                total_skills=len(result),
                by_category={
                    cat.value: sum(1 for s in result if s.category == cat)
                    for cat in SkillCategory
                    if any(s.category == cat for s in result)
                },
            )

            return result

        except SkillExtractionError:
            raise
        except Exception as e:
            logger.error("Skill extraction failed", error=str(e))
            raise SkillExtractionError(
                f"Failed to extract skills: {e}",
                detail="An unexpected error occurred during skill extraction.",
            ) from e

    # ── Layer 1: Taxonomy Pattern Matching ─────────────────────────────

    def _extract_by_pattern(self, text: str) -> list[Skill]:
        """
        Match skills from the taxonomy against the text.

        Strategy:
            1. Match multi-word phrases first (longest first) to catch
               "Ruby on Rails" before "Ruby", "React Native" before "React".
            2. Then match single-word skills with word-boundary awareness.
        """
        skills: list[Skill] = []
        text_lower = text.lower()

        # Track matched spans to avoid double-counting
        matched_spans: list[tuple[int, int]] = []

        # Phase 1: Multi-word matches (longest first)
        for phrase, defn in self._multi_word_patterns:
            for match in re.finditer(re.escape(phrase), text_lower):
                span = (match.start(), match.end())
                if not self._overlaps(span, matched_spans):
                    matched_spans.append(span)
                    skills.append(self._definition_to_skill(defn, confidence=1.0))

        # Phase 2: Single-word matches (with boundary check)
        for word, defn in self._single_word_patterns:
            if defn.requires_word_boundary:
                # Use word boundary regex for short/ambiguous terms
                pattern = r"\b" + re.escape(word) + r"\b"
                matches = list(re.finditer(pattern, text_lower))
            else:
                matches = list(re.finditer(re.escape(word), text_lower))

            for match in matches:
                span = (match.start(), match.end())
                if not self._overlaps(span, matched_spans):
                    matched_spans.append(span)
                    skills.append(self._definition_to_skill(defn, confidence=1.0))
                    break  # One match per skill is enough

        return skills

    def _build_multi_word_patterns(self) -> list[tuple[str, SkillDefinition]]:
        """Build sorted multi-word search patterns from the taxonomy."""
        return self.taxonomy.get_multi_word_entries()

    def _build_single_word_patterns(self) -> list[tuple[str, SkillDefinition]]:
        """Build sorted single-word search patterns from the taxonomy."""
        return self.taxonomy.get_single_word_entries()

    # ── Layer 2: spaCy NLP Analysis ────────────────────────────────────

    def _extract_by_nlp(self, text: str) -> list[Skill]:
        """
        Use spaCy to find skill candidates via noun phrases and NER,
        then fuzzy-match them against the taxonomy.
        """
        if self._nlp is None:
            return []

        skills: list[Skill] = []
        doc = self._nlp(text[:100000])  # Limit to 100K chars for performance

        # Extract noun phrase candidates
        candidates: set[str] = set()

        for chunk in doc.noun_chunks:
            # Clean up noun phrases — remove determiners and pronouns
            phrase = chunk.text.strip()
            if len(phrase) >= 2 and not phrase.lower().startswith(("the ", "a ", "an ", "my ", "our ")):
                candidates.add(phrase)

        # Extract named entities (ORG, PRODUCT, etc. can be tools/companies)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "LAW"):
                candidates.add(ent.text.strip())

        # Fuzzy-match candidates against taxonomy
        for candidate in candidates:
            candidate_lower = candidate.lower().strip()

            # Skip if already found by exact pattern matching
            if candidate_lower in self.taxonomy:
                continue

            # Try fuzzy match against all taxonomy entries
            best_match = self._fuzzy_match(candidate_lower)
            if best_match is not None:
                defn, score = best_match
                skills.append(self._definition_to_skill(defn, confidence=score))

        return skills

    def _fuzzy_match(
        self,
        candidate: str,
    ) -> tuple[SkillDefinition, float] | None:
        """
        Find the best fuzzy match for a candidate string in the taxonomy.

        Uses SequenceMatcher for string similarity. Returns None if
        no match exceeds the threshold.
        """
        best_defn: SkillDefinition | None = None
        best_ratio = 0.0

        for name in self.taxonomy.get_all_names_and_aliases():
            ratio = SequenceMatcher(None, candidate, name).ratio()
            if ratio > best_ratio and ratio >= self.FUZZY_THRESHOLD:
                best_ratio = ratio
                best_defn = self.taxonomy.lookup(name)

        if best_defn is not None:
            return best_defn, best_ratio
        return None

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_spacy():
        """Attempt to load spaCy with the English model."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy loaded successfully", model="en_core_web_sm")
            return nlp
        except ImportError:
            logger.info("spaCy not installed — using pattern-only extraction")
            return None
        except OSError:
            logger.info("spaCy model 'en_core_web_sm' not found — using pattern-only extraction")
            return None

    @staticmethod
    def _definition_to_skill(defn: SkillDefinition, confidence: float) -> Skill:
        """Convert a SkillDefinition to a Skill model."""
        return Skill(
            name=defn.name,
            category=defn.category,
            confidence=min(confidence, 1.0),
        )

    @staticmethod
    def _merge_skill(found: dict[str, Skill], new_skill: Skill) -> None:
        """
        Merge a new skill into the found set, keeping the higher confidence.

        Deduplicates by lowercase canonical name.
        """
        key = new_skill.name.lower()
        if key not in found or new_skill.confidence > found[key].confidence:
            found[key] = new_skill

    @staticmethod
    def _overlaps(span: tuple[int, int], existing: list[tuple[int, int]]) -> bool:
        """Check if a span overlaps with any existing matched span."""
        for start, end in existing:
            if span[0] < end and span[1] > start:
                return True
        return False

    # ── Public Utilities ───────────────────────────────────────────────

    def extract_from_both(
        self,
        resume_text: str,
        job_description: str,
    ) -> tuple[list[Skill], list[Skill]]:
        """
        Extract skills from both resume and job description.

        Convenience method that returns both lists in a single call.

        Returns:
            Tuple of (resume_skills, job_skills).
        """
        resume_skills = self.extract_skills(resume_text, source="resume")
        job_skills = self.extract_skills(job_description, source="job_description")
        return resume_skills, job_skills

    def get_taxonomy_size(self) -> int:
        """Return the number of skills in the taxonomy."""
        return self.taxonomy.size
