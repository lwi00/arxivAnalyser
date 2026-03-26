"""Tests for section classification: SectionType.from_heading and SectionClassifier._fuzzy_match."""

from __future__ import annotations

import pytest

from src.models import RawSection, SectionType
from src.section_classifier import SectionClassifier


@pytest.fixture
def classifier():
    return SectionClassifier(use_llm=False)


# --- SectionType.from_heading (direct keyword match) ---


class TestFromHeading:
    def test_exact_related_work(self):
        assert SectionType.from_heading("Related Work") == SectionType.RELATED_WORK

    def test_case_insensitive(self):
        assert SectionType.from_heading("RELATED WORKS") == SectionType.RELATED_WORK

    def test_literature_review(self):
        assert SectionType.from_heading("Literature Review") == SectionType.RELATED_WORK

    def test_prior_work(self):
        assert SectionType.from_heading("Prior Work") == SectionType.RELATED_WORK

    def test_unrelated_heading_returns_other(self):
        assert SectionType.from_heading("Introduction") == SectionType.OTHER

    def test_empty_heading_returns_other(self):
        assert SectionType.from_heading("") == SectionType.OTHER


# --- SectionClassifier._fuzzy_match ---


class TestFuzzyMatch:
    def test_plain_related_work(self, classifier):
        result = classifier._fuzzy_match("Related Work")
        assert result is not None
        assert result[0] == SectionType.RELATED_WORK

    def test_numbered_prefix_stripped(self, classifier):
        result = classifier._fuzzy_match("3. Related Work")
        assert result is not None
        assert result[0] == SectionType.RELATED_WORK

    def test_roman_numeral_prefix_stripped(self, classifier):
        result = classifier._fuzzy_match("iv. Related Work")
        assert result is not None
        assert result[0] == SectionType.RELATED_WORK

    def test_decimal_prefix_stripped(self, classifier):
        result = classifier._fuzzy_match("3.1 Related Work")
        assert result is not None
        assert result[0] == SectionType.RELATED_WORK

    def test_no_corruption_of_normal_headings(self, classifier):
        """Regression test: lstrip('0123456789ivx.') would corrupt headings
        starting with i/v/x characters. The regex fix should not."""
        result = classifier._fuzzy_match("overview of methods")
        # "overview" should NOT be corrupted to "erview" or similar
        assert result is None  # not related work

    def test_no_corruption_visual(self, classifier):
        """Headings starting with 'v' should not be stripped."""
        result = classifier._fuzzy_match("visual representations")
        assert result is None

    def test_existing_approaches(self, classifier):
        result = classifier._fuzzy_match("existing approaches")
        assert result is not None
        assert result[0] == SectionType.RELATED_WORK

    def test_background_and_related(self, classifier):
        result = classifier._fuzzy_match("Background and Related Work")
        assert result is not None
        assert result[0] == SectionType.RELATED_WORK

    def test_no_match_returns_none(self, classifier):
        assert classifier._fuzzy_match("Conclusion") is None

    def test_empty_heading_returns_none(self, classifier):
        assert classifier._fuzzy_match("") is None


# --- classify_sections end-to-end (no LLM) ---


class TestClassifySections:
    def test_direct_match_used_first(self, classifier):
        sections = [RawSection(heading="Related Work", content="Some content.", order=0)]
        classified = classifier.classify_sections(sections)
        assert len(classified) == 1
        assert classified[0].section_type == SectionType.RELATED_WORK
        assert classified[0].confidence == 1.0

    def test_unknown_heading_defaults_to_other(self, classifier):
        sections = [RawSection(heading="Experiments", content="We ran tests.", order=0)]
        classified = classifier.classify_sections(sections)
        assert len(classified) == 1
        assert classified[0].section_type == SectionType.OTHER
