"""
SkillForge AI — PDF Text Extraction.

Uses PyMuPDF (fitz) to extract text from uploaded PDF resumes.
Handles multi-page documents, encoding issues, and common failure
modes with structured error reporting.

Usage:
    from src.skillforge.data.pdf_parser import PDFParser

    parser = PDFParser()
    result = parser.extract_text(pdf_bytes)
    print(result.text)
    print(f"Pages: {result.page_count}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fitz  # PyMuPDF

from src.skillforge.utils.exceptions import PDFParsingError
from src.skillforge.utils.logging import logger


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""

    text: str
    page_count: int
    pages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """
    Extracts text content from PDF files using PyMuPDF.

    Designed for resume PDFs which may use varied layouts,
    fonts, and structures. Falls back gracefully when pages
    can't be parsed and collects warnings for the caller.
    """

    # Minimum characters per page to consider extraction successful
    MIN_CHARS_PER_PAGE = 10

    # Maximum file size to accept (20 MB)
    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

    def extract_text(self, pdf_bytes: bytes) -> PDFExtractionResult:
        """
        Extract text from a PDF file provided as raw bytes.

        Args:
            pdf_bytes: Raw bytes of the PDF file.

        Returns:
            PDFExtractionResult with extracted text, page count, and any warnings.

        Raises:
            PDFParsingError: If the file is not a valid PDF, is empty,
                or text extraction completely fails.
        """
        self._validate_input(pdf_bytes)

        warnings: list[str] = []
        pages: list[str] = []
        metadata: dict = {}

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error("Failed to open PDF document", error=str(e))
            raise PDFParsingError(
                f"Could not open PDF: {e}",
                detail="The file may be corrupted or not a valid PDF.",
            ) from e

        try:
            # Extract document metadata
            metadata = self._extract_metadata(doc)
            page_count = len(doc)

            if page_count == 0:
                raise PDFParsingError(
                    "PDF contains no pages",
                    detail="The uploaded file appears to be an empty PDF.",
                )

            logger.info("Extracting text from PDF", pages=page_count)

            # Extract text from each page
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")

                    if len(page_text.strip()) < self.MIN_CHARS_PER_PAGE:
                        warning = f"Page {page_num + 1}: Very little text extracted ({len(page_text.strip())} chars). Page may contain images or scanned content."
                        warnings.append(warning)
                        logger.warning(warning)

                    pages.append(page_text)

                except Exception as e:
                    warning = f"Page {page_num + 1}: Failed to extract text ({e})"
                    warnings.append(warning)
                    pages.append("")
                    logger.warning(warning)

            # Combine all page text
            full_text = "\n\n".join(pages)

            if not full_text.strip():
                raise PDFParsingError(
                    "No text could be extracted from the PDF",
                    detail="The PDF may contain only images or scanned content. "
                    "Try a PDF with selectable text, or paste your resume text directly.",
                )

            logger.info(
                "PDF text extraction complete",
                pages=page_count,
                chars=len(full_text),
                warnings=len(warnings),
            )

            return PDFExtractionResult(
                text=full_text,
                page_count=page_count,
                pages=pages,
                warnings=warnings,
                metadata=metadata,
            )

        finally:
            doc.close()

    def extract_text_from_path(self, file_path: str) -> PDFExtractionResult:
        """
        Extract text from a PDF file on disk.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFExtractionResult with extracted text.

        Raises:
            PDFParsingError: If the file can't be read or parsed.
        """
        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
        except FileNotFoundError:
            raise PDFParsingError(
                f"PDF file not found: {file_path}",
                detail="Check that the file path is correct.",
            )
        except PermissionError:
            raise PDFParsingError(
                f"Permission denied reading: {file_path}",
                detail="Check file permissions.",
            )
        except Exception as e:
            raise PDFParsingError(
                f"Error reading PDF file: {e}",
                detail="An unexpected error occurred while reading the file.",
            ) from e

        return self.extract_text(pdf_bytes)

    def _validate_input(self, pdf_bytes: bytes) -> None:
        """Validate the input bytes before attempting to parse."""
        if not pdf_bytes:
            raise PDFParsingError(
                "Empty input: no PDF data provided",
                detail="The uploaded file appears to be empty.",
            )

        if len(pdf_bytes) > self.MAX_FILE_SIZE_BYTES:
            size_mb = len(pdf_bytes) / (1024 * 1024)
            raise PDFParsingError(
                f"PDF too large: {size_mb:.1f} MB (max: {self.MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB)",
                detail="Please upload a smaller file.",
            )

        # Check for PDF magic bytes (%PDF)
        if not pdf_bytes[:5].startswith(b"%PDF"):
            raise PDFParsingError(
                "File does not appear to be a valid PDF",
                detail="The file header does not match the PDF format. "
                "Make sure you're uploading a .pdf file.",
            )

    def _extract_metadata(self, doc: fitz.Document) -> dict:
        """Extract available metadata from the PDF document."""
        try:
            meta = doc.metadata or {}
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "page_count": len(doc),
            }
        except Exception:
            return {"page_count": len(doc)}
