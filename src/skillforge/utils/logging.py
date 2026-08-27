"""
SkillForge AI — Structured Logging Configuration.

Configures Loguru with structured output, rotation, and appropriate
formatting for both console and file outputs. All modules should
import `logger` from this module rather than using print() or
stdlib logging.

Usage:
    from src.skillforge.utils.logging import setup_logging, logger

    setup_logging("DEBUG")
    logger.info("Processing resume", filename="resume.pdf", pages=3)
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path | None = None,
    enable_file_logging: bool = False,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Minimum log level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to './logs'.
        enable_file_logging: Whether to write logs to a file in addition to stderr.
    """
    # Remove any default handlers to avoid duplicate output
    logger.remove()

    # Console handler — human-readable format with color
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler — structured JSON for machine parsing (optional)
    if enable_file_logging:
        log_path = Path(log_dir or "logs")
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path / "skillforge_{time:YYYY-MM-DD}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=False,  # Don't leak variable values in production logs
        )

    logger.info(
        "Logging initialized",
        level=log_level,
        file_logging=enable_file_logging,
    )


# Re-export logger so modules can do: from src.skillforge.utils.logging import logger
__all__ = ["setup_logging", "logger"]
