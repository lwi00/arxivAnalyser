r"""LaTeX source parser for extracting document sections.

This module parses LaTeX source files to extract structured sections from
academic papers. It handles the many quirks of real-world LaTeX:

- Multi-file projects with ``\input{}`` and ``\include{}`` directives
- Various document classes (article, IEEEtran, LNCS, ACM, NeurIPS, etc.)
- Custom section commands and numbering
- Nested sections and subsections (flattened to top-level)
- Abstract environments (``\begin{abstract}...\end{abstract}``)
- LaTeX command stripping for clean text output

The parser uses regex-based extraction rather than full LaTeX parsing
(e.g., TexSoup) because real ArXiv papers frequently contain non-standard
LaTeX that breaks strict parsers. The regex approach is more robust for
the messy reality of ArXiv sources.

Typical usage:
    ```python
    parser = LatexParser()
    sections = parser.parse_source_directory("/path/to/extracted/source")
    for section in sections:
        print(f"{section.heading}: {len(section.content)} chars")
    ```
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from pylatexenc.latex2text import LatexNodes2Text

from src.models import RawSection

logger = logging.getLogger(__name__)

# Regex patterns for LaTeX section commands
# Matches: \section{Title}, \section*{Title}, \section[short]{Title}
SECTION_PATTERN = re.compile(
    r"\\(section|subsection|subsubsection)\*?"
    r"(?:\[[^\]]*\])?"  # Optional short title
    r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Title with possible nested braces
    re.MULTILINE,
)

# Pattern for abstract environment
ABSTRACT_PATTERN = re.compile(
    r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
    re.DOTALL,
)

# Pattern for \input{} and \include{} directives
INPUT_PATTERN = re.compile(
    r"\\(?:input|include)\{([^}]+)\}",
)

# Pattern for LaTeX comments
COMMENT_PATTERN = re.compile(r"(?<!\\)%.*$", re.MULTILINE)

# Common file encodings to try
ENCODINGS = ["utf-8", "latin-1", "ascii", "cp1252"]


class LatexParser:
    """Parses LaTeX source files to extract document sections.

    This parser is designed for robustness over strict correctness. It
    handles the many variations of LaTeX formatting found in ArXiv papers,
    including non-standard macros, encoding issues, and multi-file projects.

    Attributes:
        max_file_size: Maximum file size to process (bytes).
        latex2text: Converter for stripping LaTeX commands.
    """

    def __init__(self, max_file_size: int = 10_000_000):
        """Initialize the parser.

        Args:
            max_file_size: Max file size in bytes to parse. Files larger
                than this are skipped to avoid memory issues.
        """
        self.max_file_size = max_file_size
        self.latex2text = LatexNodes2Text()

    def parse_source_directory(self, source_dir: str | Path) -> list[RawSection]:
        """Parse all LaTeX files in a source directory.

        This is the main entry point. It finds the root .tex file,
        resolves all \\input/\\include directives, and extracts sections
        from the assembled document.

        Args:
            source_dir: Path to the extracted source directory.

        Returns:
            List of RawSection objects in document order.
            Returns empty list if parsing fails.
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            logger.warning("Source directory does not exist: %s", source_dir)
            return []

        # Find the root tex file
        root_tex = self._find_root_tex(source_dir)
        if root_tex is None:
            logger.warning("No root .tex file found in %s", source_dir)
            return []

        # Read and assemble the full document (resolving \input directives)
        full_text = self._assemble_document(root_tex, source_dir)
        if not full_text:
            return []

        # Remove comments
        full_text = COMMENT_PATTERN.sub("", full_text)

        # Extract sections
        sections = self._extract_sections(full_text)

        logger.debug(
            "Parsed %d sections from %s", len(sections), source_dir.name
        )
        return sections

    def _find_root_tex(self, source_dir: Path) -> Optional[Path]:
        """Find the root .tex file in a source directory.

        Heuristics (in priority order):
        1. File containing ``\\documentclass``
        2. File named ``main.tex``
        3. File named ``paper.tex``
        4. The only ``.tex`` file (if there's just one)
        5. The largest ``.tex`` file

        Args:
            source_dir: Directory to search.

        Returns:
            Path to the root .tex file, or None if not found.
        """
        tex_files = list(source_dir.rglob("*.tex"))
        if not tex_files:
            return None

        if len(tex_files) == 1:
            return tex_files[0]

        # Look for \documentclass
        for tex_file in tex_files:
            try:
                content = self._read_file(tex_file)
                if content and r"\documentclass" in content:
                    return tex_file
            except Exception:
                continue

        # Try common names
        for name in ["main.tex", "paper.tex", "manuscript.tex", "article.tex"]:
            candidate = source_dir / name
            if candidate.exists():
                return candidate

        # Fallback: largest file
        return max(tex_files, key=lambda f: f.stat().st_size)

    def _assemble_document(self, root_path: Path, base_dir: Path) -> str:
        """Read the root .tex file and recursively resolve \\input directives.

        Args:
            root_path: Path to the root .tex file.
            base_dir: Base directory for resolving relative paths.

        Returns:
            The fully assembled document text.
        """
        content = self._read_file(root_path)
        if not content:
            return ""

        # Recursively resolve \input{} and \include{}
        resolved = self._resolve_inputs(content, base_dir, depth=0, max_depth=10)
        return resolved

    def _resolve_inputs(
        self, content: str, base_dir: Path, depth: int, max_depth: int
    ) -> str:
        """Recursively replace \\input{} and \\include{} with file contents.

        Args:
            content: Current LaTeX content.
            base_dir: Base directory for relative paths.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth to prevent infinite loops.

        Returns:
            Content with all input directives resolved.
        """
        if depth >= max_depth:
            return content

        def replace_input(match):
            filename = match.group(1).strip()
            # Add .tex extension if missing
            if not filename.endswith(".tex"):
                filename += ".tex"

            input_path = base_dir / filename
            if not input_path.exists():
                # Try searching subdirectories
                candidates = list(base_dir.rglob(filename))
                if candidates:
                    input_path = candidates[0]
                else:
                    logger.debug("Input file not found: %s", filename)
                    return ""

            input_content = self._read_file(input_path)
            if input_content:
                return self._resolve_inputs(
                    input_content, input_path.parent, depth + 1, max_depth
                )
            return ""

        return INPUT_PATTERN.sub(replace_input, content)

    def _extract_sections(self, text: str) -> list[RawSection]:
        """Extract sections from assembled LaTeX text.

        Handles both \\section{} commands and \\begin{abstract} environments.
        Subsections and subsubsections are merged into their parent section.

        Args:
            text: Full LaTeX document text.

        Returns:
            List of RawSection objects in document order.
        """
        sections: list[RawSection] = []

        # 1. Extract abstract
        abstract_match = ABSTRACT_PATTERN.search(text)
        if abstract_match:
            abstract_text = self._clean_latex(abstract_match.group(1))
            if abstract_text.strip():
                sections.append(
                    RawSection(heading="Abstract", content=abstract_text.strip(), order=0)
                )

        # 2. Find all \section{} commands and their positions
        section_matches = list(SECTION_PATTERN.finditer(text))

        # Filter to top-level sections only (section, not subsection)
        top_level_matches = [
            m for m in section_matches if m.group(1) == "section"
        ]

        if not top_level_matches:
            # No \section{} found — try alternative patterns
            top_level_matches = self._find_alternative_sections(text)

        # 3. Extract content between section headers
        for i, match in enumerate(top_level_matches):
            heading = self._clean_latex(match.group(2) if hasattr(match, 'group') and match.lastindex >= 2 else match.group(1))
            start = match.end()
            end = (
                top_level_matches[i + 1].start()
                if i + 1 < len(top_level_matches)
                else len(text)
            )

            content = text[start:end]

            # Remove subsection headers but keep their content
            content = re.sub(
                r"\\(?:subsection|subsubsection)\*?(?:\[[^\]]*\])?\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                "\n",
                content,
            )

            cleaned = self._clean_latex(content)
            if cleaned.strip():
                sections.append(
                    RawSection(
                        heading=heading.strip(),
                        content=cleaned.strip(),
                        order=len(sections),
                    )
                )

        return sections

    def _find_alternative_sections(self, text: str) -> list:
        """Find sections using alternative LaTeX patterns.

        Some document classes (IEEEtran, LNCS) use non-standard section commands.
        This method handles common alternatives.

        Args:
            text: LaTeX document text.

        Returns:
            List of regex match objects for section boundaries.
        """
        # Try numbered sections like "1. Introduction" or "I. INTRODUCTION"
        alt_pattern = re.compile(
            r"^(?:(?:\d+|[IVX]+)\.?\s+)([A-Z][A-Za-z\s]+)$",
            re.MULTILINE,
        )
        matches = list(alt_pattern.finditer(text))
        if matches:
            return matches

        return []

    def _clean_latex(self, text: str) -> str:
        """Convert LaTeX markup to plain text.

        Uses pylatexenc for robust conversion, with regex fallbacks for
        commands that pylatexenc doesn't handle well.

        Args:
            text: LaTeX text to clean.

        Returns:
            Plain text with LaTeX commands removed.
        """
        try:
            # First pass: pylatexenc
            cleaned = self.latex2text.latex_to_text(text)
        except Exception:
            # Fallback: manual regex stripping
            cleaned = text

        # Additional cleanup
        # Remove remaining commands like \cite{}, \ref{}, \label{}
        cleaned = re.sub(r"\\(?:cite|ref|label|eqref|autoref)\{[^}]*\}", "", cleaned)
        # Remove \textbf{}, \textit{}, \emph{} but keep content
        cleaned = re.sub(r"\\(?:textbf|textit|emph|textsc|textrm)\{([^}]*)\}", r"\1", cleaned)
        # Remove math environments but keep content
        cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)
        # Remove remaining backslash commands
        cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})?", "", cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        return cleaned.strip()

    def _read_file(self, path: Path) -> Optional[str]:
        """Read a file trying multiple encodings.

        Args:
            path: Path to the file to read.

        Returns:
            File contents as string, or None if all encodings fail.
        """
        if path.stat().st_size > self.max_file_size:
            logger.warning("File too large, skipping: %s (%d bytes)", path, path.stat().st_size)
            return None

        for encoding in ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue

        logger.warning("Could not decode file with any encoding: %s", path)
        return None
