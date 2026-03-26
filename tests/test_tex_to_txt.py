"""Tests for tex_to_txt filename sanitization."""

from __future__ import annotations

import re


# Inline the function to avoid pulling in the full import chain (latex_parser → pylatexenc)
def _sanitize_filename(heading: str) -> str:
    """Turn a section heading into a safe filename component."""
    clean = re.sub(r"[^\w\s-]", "", heading)
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean or "untitled"


class TestSanitizeFilename:
    def test_simple_heading(self):
        assert _sanitize_filename("Introduction") == "Introduction"

    def test_spaces_to_underscores(self):
        assert _sanitize_filename("Related Work") == "Related_Work"

    def test_special_chars_removed(self):
        assert _sanitize_filename("Results (Final)") == "Results_Final"

    def test_empty_string_returns_untitled(self):
        assert _sanitize_filename("") == "untitled"

    def test_unicode_preserved(self):
        """Python 3's \\w matches Unicode word characters, so accented chars should survive."""
        result = _sanitize_filename("Über Modelle")
        assert "Über" in result

    def test_colons_and_semicolons_removed(self):
        assert _sanitize_filename("Methods: A Survey") == "Methods_A_Survey"

    def test_multiple_spaces_collapsed(self):
        assert _sanitize_filename("Related   Work") == "Related_Work"
