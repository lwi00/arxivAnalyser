"""Data models for the ArXiv section extraction pipeline.

This module defines the core data structures used throughout the pipeline,
including paper metadata, extracted sections, and processing status tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SectionType(str, Enum):
    """Standardized section types for classification.

    These represent the canonical sections that papers are mapped to.
    Papers may use different headings, but they are normalized to these types.
    """

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHOD = "method"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    OTHER = "other"

    @classmethod
    def from_heading(cls, heading: str) -> SectionType:
        """Attempt to classify a section heading into a standard type.

        Uses keyword matching as a first-pass heuristic. Handles common
        variations like 'methodology', 'experimental setup', 'findings', etc.

        Args:
            heading: The raw section heading text from the paper.

        Returns:
            The matched SectionType, or SectionType.OTHER if no match.
        """
        heading_lower = heading.lower().strip()

        abstract_keywords = {"abstract"}
        intro_keywords = {
            "introduction",
            "background",
            "motivation",
            "overview",
            "preliminaries",
            "preliminary",
        }
        method_keywords = {
            "method",
            "methodology",
            "approach",
            "framework",
            "model",
            "architecture",
            "system",
            "proposed",
            "design",
            "implementation",
            "experimental setup",
            "setup",
            "formulation",
            "problem formulation",
            "technique",
        }
        results_keywords = {
            "result",
            "experiment",
            "evaluation",
            "finding",
            "analysis",
            "empirical",
            "performance",
            "comparison",
            "benchmark",
            "ablation",
            "discussion",
        }
        conclusion_keywords = {
            "conclusion",
            "summary",
            "future work",
            "concluding",
            "limitations",
            "limitation",
            "broader impact",
        }

        for keyword in abstract_keywords:
            if keyword in heading_lower:
                return cls.ABSTRACT
        for keyword in intro_keywords:
            if keyword in heading_lower:
                return cls.INTRODUCTION
        for keyword in method_keywords:
            if keyword in heading_lower:
                return cls.METHOD
        for keyword in results_keywords:
            if keyword in heading_lower:
                return cls.RESULTS
        for keyword in conclusion_keywords:
            if keyword in heading_lower:
                return cls.CONCLUSION

        return cls.OTHER


class ExtractionMethod(str, Enum):
    """How the sections were extracted from the paper."""

    LATEX_PARSED = "latex_parsed"
    PDF_EXTRACTED = "pdf_extracted"
    LLM_CLASSIFIED = "llm_classified"


class ProcessingStatus(str, Enum):
    """Status of paper processing."""

    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PARSED = "parsed"
    FAILED_DOWNLOAD = "failed_download"
    FAILED_PARSE = "failed_parse"
    COMPLETED = "completed"


@dataclass
class RawSection:
    """A section as extracted from the source, before classification.

    Attributes:
        heading: The original heading text (e.g., '3. Our Approach').
        content: The full text content of the section.
        order: Position of this section in the paper (0-indexed).
    """

    heading: str
    content: str
    order: int


@dataclass
class ClassifiedSection:
    """A section after classification into a standard type.

    Attributes:
        section_type: The standardized section category.
        original_heading: The original heading from the paper.
        content: The full text content.
        confidence: Classification confidence (1.0 for heuristic matches).
        method: How this section was classified.
    """

    section_type: SectionType
    original_heading: str
    content: str
    confidence: float = 1.0
    method: ExtractionMethod = ExtractionMethod.LATEX_PARSED


@dataclass
class PaperMetadata:
    """Metadata for an ArXiv paper.

    Attributes:
        arxiv_id: The ArXiv identifier (e.g., '2301.12345').
        title: Paper title.
        authors: List of author names.
        categories: ArXiv categories (e.g., ['cs.CL', 'cs.AI']).
        primary_category: Primary ArXiv category.
        published: Publication date as ISO string.
        updated: Last update date as ISO string.
        abstract: Paper abstract text.
        doi: DOI if available.
    """

    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abstract: str
    doi: Optional[str] = None


@dataclass
class PaperRecord:
    """Complete processed paper record for output to Parquet.

    This is the final flattened structure that gets written to the dataset.
    Each field maps to a column in the Parquet file.

    Attributes:
        arxiv_id: The ArXiv identifier.
        title: Paper title.
        authors: Semicolon-separated author list.
        categories: Comma-separated category list.
        primary_category: Primary ArXiv category.
        published: Publication date.
        updated: Last update date.
        abstract: Abstract text.
        introduction: Introduction text (empty string if not found).
        method: Method/approach section text.
        results: Results/experiments section text.
        conclusion: Conclusion text.
        extraction_method: How the paper was processed.
        extraction_success: Whether all standard sections were found.
        sections_found: Comma-separated list of sections that were extracted.
        raw_section_count: Total number of sections in the original paper.
    """

    arxiv_id: str
    title: str
    authors: str
    categories: str
    primary_category: str
    published: str
    updated: str
    abstract: str
    introduction: str = ""
    method: str = ""
    results: str = ""
    conclusion: str = ""
    extraction_method: str = ""
    extraction_success: bool = False
    sections_found: str = ""
    raw_section_count: int = 0
