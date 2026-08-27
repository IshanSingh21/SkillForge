"""
SkillForge AI — Text Preprocessing.

Cleans raw text extracted from PDFs and detects resume sections.
Handles common OCR artifacts, excessive whitespace, encoding issues,
and provides section-level structure for downstream processing.

Usage:
    from src.skillforge.data.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(raw_text)
    sections = preprocessor.extract_sections(cleaned)
"""

from __future__ import annotations

import re
import unicodedata

from src.skillforge.models.resume import ResumeSection
from src.skillforge.utils.exceptions import PreprocessingError
from src.skillforge.utils.logging import logger


class TextPreprocessor:
    """
    Cleans and structures raw resume text.

    Applies a pipeline of normalization steps and uses pattern matching
    to identify common resume sections (Education, Experience, Skills, etc.).
    """

    # Common resume section headings (case-insensitive patterns)
    SECTION_PATTERNS: list[re.Pattern] = [
        re.compile(
            r"^\s*(?:PROFESSIONAL\s+)?(?:WORK\s+)?EXPERIENCE\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*EDUCATION(?:\s+(?:AND|&)\s+\w+)?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:TECHNICAL\s+|KEY\s+|CORE\s+|RELEVANT\s+)?SKILLS?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:PERSONAL\s+|CAREER\s+)?(?:SUMMARY|OBJECTIVE|PROFILE|ABOUT\s*ME)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:KEY\s+|PERSONAL\s+)?PROJECTS?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*CERTIFICATIONS?\s*(?:AND\s+LICENSES?)?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:PROFESSIONAL\s+)?PUBLICATIONS?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:VOLUNTEER(?:ING)?|COMMUNITY)\s*(?:EXPERIENCE|WORK)?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*AWARDS?\s*(?:AND\s+(?:HONORS?|ACHIEVEMENTS?))?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:ADDITIONAL\s+)?INTERESTS?\s*(?:AND\s+HOBBIES)?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*LANGUAGES?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*REFERENCES?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*CONTACT\s*(?:INFORMATION|DETAILS|INFO)?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]

    def clean_text(self, raw_text: str) -> str:
        """
        Apply a pipeline of cleaning steps to raw extracted text.

        Steps:
            1. Normalize Unicode characters
            2. Fix encoding artifacts
            3. Normalize whitespace
            4. Remove excessive blank lines
            5. Strip leading/trailing whitespace

        Args:
            raw_text: Raw text from PDF extraction.

        Returns:
            Cleaned text ready for further processing.

        Raises:
            PreprocessingError: If input is empty or cleaning fails.
        """
        if not raw_text or not raw_text.strip():
            raise PreprocessingError(
                "Cannot clean empty text",
                detail="No text was provided for preprocessing.",
            )

        try:
            text = raw_text

            # Step 1: Normalize Unicode (NFC form — composed characters)
            text = unicodedata.normalize("NFC", text)

            # Step 2: Fix common encoding artifacts
            text = self._fix_encoding_artifacts(text)

            # Step 3: Normalize whitespace
            # Replace tabs with spaces
            text = text.replace("\t", "    ")
            # Collapse multiple spaces into one (but preserve newlines)
            text = re.sub(r"[^\S\n]+", " ", text)

            # Step 4: Normalize line endings and reduce excessive blank lines
            text = re.sub(r"\r\n", "\n", text)       # Normalize CRLF → LF
            text = re.sub(r"\n{4,}", "\n\n\n", text)  # Max 2 blank lines

            # Step 5: Strip leading/trailing whitespace from each line
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(lines)

            # Step 6: Strip overall leading/trailing whitespace
            text = text.strip()

            logger.debug(
                "Text cleaned",
                original_len=len(raw_text),
                cleaned_len=len(text),
                reduction_pct=f"{(1 - len(text) / max(len(raw_text), 1)) * 100:.1f}%",
            )

            return text

        except PreprocessingError:
            raise
        except Exception as e:
            logger.error("Text cleaning failed", error=str(e))
            raise PreprocessingError(
                f"Text cleaning failed: {e}",
                detail="An unexpected error occurred during text preprocessing.",
            ) from e

    def extract_sections(self, text: str) -> list[ResumeSection]:
        """
        Detect and extract named sections from resume text.

        Uses regex patterns to identify common section headings and
        extracts the content between them.

        Args:
            text: Cleaned resume text.

        Returns:
            List of ResumeSection objects, ordered by position in text.
        """
        if not text or not text.strip():
            return []

        # Find all section heading matches
        matches: list[tuple[str, int, int]] = []  # (title, start, end)

        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                title = match.group().strip()
                matches.append((title, match.start(), match.end()))

        if not matches:
            logger.info("No standard sections detected in text")
            return []

        # Sort by position in text
        matches.sort(key=lambda m: m[1])

        # Build sections — each section's content runs from after its heading
        # to the start of the next heading (or end of text)
        sections: list[ResumeSection] = []

        for i, (title, start, end) in enumerate(matches):
            # Content starts after the heading line
            content_start = end

            # Content ends at the start of the next section (or end of text)
            if i + 1 < len(matches):
                content_end = matches[i + 1][1]
            else:
                content_end = len(text)

            content = text[content_start:content_end].strip()

            if content:  # Only include sections with actual content
                sections.append(
                    ResumeSection(
                        title=self._normalize_section_title(title),
                        content=content,
                        start_index=start,
                        end_index=content_end,
                    )
                )

        logger.info(
            "Sections extracted",
            count=len(sections),
            titles=[s.title for s in sections],
        )

        return sections

    def _fix_encoding_artifacts(self, text: str) -> str:
        """Fix common encoding artifacts from PDF extraction."""
        replacements = {
            "\u2018": "'",   # Left single quote
            "\u2019": "'",   # Right single quote
            "\u201c": '"',   # Left double quote
            "\u201d": '"',   # Right double quote
            "\u2013": "-",   # En dash
            "\u2014": "-",   # Em dash
            "\u2026": "...", # Ellipsis
            "\u00a0": " ",   # Non-breaking space
            "\u200b": "",    # Zero-width space
            "\ufeff": "",    # BOM
            "\u00ad": "",    # Soft hyphen
            "\uf0b7": "- ",  # Bullet point (common in PDFs)
            "\uf0a7": "- ",  # Another bullet variant
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _normalize_section_title(self, title: str) -> str:
        """Normalize section titles to a consistent format."""
        title = title.strip()
        # Title-case the section heading
        title = title.title()
        # Fix common casing quirks
        title = title.replace("And", "and").replace("Of", "of")
        return title
