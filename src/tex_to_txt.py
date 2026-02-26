"""Convert LaTeX source files to plain text, preserving section structure.

Uses LatexParser.parse_source_directory() to extract individual sections
(Abstract, Introduction, Method, etc.) and writes each as a separate .txt
file inside a per-paper subdirectory.

Output structure:
    data/txt/{arxiv_id}/
        00_Abstract.txt
        01_Introduction.txt
        02_Background.txt
        ...

Typical usage:
    # Single paper
    convert_tex_to_txt(Path("data/sources/1706.03762"), Path("data/txt"), "1706.03762")

    # Batch conversion
    results = batch_convert_tex_to_txt(Path("data/sources"), Path("data/txt"))
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.latex_parser import LatexParser

logger = logging.getLogger(__name__)


def _sanitize_filename(heading: str) -> str:
    """Turn a section heading into a safe filename component."""
    # Keep only alphanumeric, spaces, hyphens
    clean = re.sub(r"[^\w\s-]", "", heading)
    # Collapse whitespace to underscores
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean or "untitled"


def convert_tex_to_txt(
    source_dir: Path,
    output_dir: Path,
    arxiv_id: str,
    max_file_size: int = 10_000_000,
) -> bool:
    """Convert a single paper's LaTeX source to per-section .txt files.

    Parses the LaTeX source into RawSection objects (already cleaned of
    LaTeX commands) and writes each section as a separate .txt file.

    Args:
        source_dir: Path to the paper's extracted source directory.
        output_dir: Parent directory; files go into output_dir/{arxiv_id}/.
        arxiv_id: ArXiv identifier, used as the subdirectory name.
        max_file_size: Maximum file size in bytes for the parser.

    Returns:
        True if at least one section was written, False otherwise.
    """
    parser = LatexParser(max_file_size=max_file_size)
    sections = parser.parse_source_directory(source_dir)

    if not sections:
        logger.warning("No sections extracted for %s", arxiv_id)
        return False

    paper_dir = Path(output_dir) / arxiv_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    for section in sections:
        filename = f"{section.order:02d}_{_sanitize_filename(section.heading)}.txt"
        filepath = paper_dir / filename
        filepath.write_text(section.content, encoding="utf-8")

    logger.info(
        "Wrote %d sections for %s (%s)",
        len(sections), arxiv_id, paper_dir,
    )
    return True


def batch_convert_tex_to_txt(
    sources_dir: Path,
    output_dir: Path,
    max_file_size: int = 10_000_000,
) -> dict:
    """Convert all papers in a sources directory to per-section .txt files.

    Iterates over subdirectories in sources_dir, treating each as a
    paper's extracted LaTeX source.

    Args:
        sources_dir: Parent directory containing per-paper subdirectories.
        output_dir: Directory where per-paper subdirectories will be created.
        max_file_size: Maximum file size in bytes for the parser.

    Returns:
        Summary dict with keys: "succeeded", "failed", "skipped", "total".
    """
    sources_dir = Path(sources_dir)
    output_dir = Path(output_dir)

    if not sources_dir.exists():
        logger.error("Sources directory does not exist: %s", sources_dir)
        return {"succeeded": 0, "failed": 0, "skipped": 0, "total": 0}

    subdirs = sorted(
        [d for d in sources_dir.iterdir() if d.is_dir()]
    )

    succeeded = 0
    failed = 0
    skipped = 0

    for paper_dir in subdirs:
        arxiv_id = paper_dir.name
        paper_out = output_dir / arxiv_id

        # Skip if already converted
        if paper_out.exists() and any(paper_out.glob("*.txt")):
            skipped += 1
            continue

        try:
            ok = convert_tex_to_txt(paper_dir, output_dir, arxiv_id, max_file_size)
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            logger.warning("Conversion failed for %s: %s", arxiv_id, e)
            failed += 1

    total = succeeded + failed + skipped
    logger.info(
        "Batch conversion complete: %d succeeded, %d failed, %d skipped (total: %d)",
        succeeded, failed, skipped, total,
    )
    return {
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total": total,
    }
