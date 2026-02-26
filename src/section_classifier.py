"""Section classifier: maps raw sections to standard types.

This module classifies extracted sections (from LaTeX or PDF) into the
standard categories: abstract, introduction, method, results, conclusion.

Classification strategy (in priority order):
1. **Exact heuristic match** — direct keyword matching on headings.
   Handles ~80% of CS papers with high confidence.
2. **Fuzzy heuristic match** — looser keyword matching and content-based
   heuristics for non-standard headings.
3. **LLM fallback** — sends ambiguous sections to Claude for classification.
   Used only when heuristics fail and `use_llm_fallback=True`.

The goal is to minimize LLM calls (for cost) while maximizing accuracy.

Typical usage:
    ```python
    classifier = SectionClassifier(use_llm=True)
    classified = classifier.classify_sections(raw_sections)
    ```
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.models import (
    ClassifiedSection,
    ExtractionMethod,
    RawSection,
    SectionType,
)

logger = logging.getLogger(__name__)

# Extended keyword mappings for fuzzy matching
# Each entry is (keywords_in_heading, section_type, confidence)
FUZZY_RULES: list[tuple[list[str], SectionType, float]] = [
    # Abstract
    (["abstract"], SectionType.ABSTRACT, 1.0),
    # Introduction
    (["introduction"], SectionType.INTRODUCTION, 1.0),
    (["background"], SectionType.INTRODUCTION, 0.8),
    (["motivation"], SectionType.INTRODUCTION, 0.8),
    (["overview"], SectionType.INTRODUCTION, 0.7),
    (["preliminaries", "preliminary"], SectionType.INTRODUCTION, 0.7),
    (["related work"], SectionType.INTRODUCTION, 0.7),
    # Method
    (["method"], SectionType.METHOD, 1.0),
    (["methodology"], SectionType.METHOD, 1.0),
    (["approach"], SectionType.METHOD, 0.9),
    (["proposed"], SectionType.METHOD, 0.9),
    (["framework"], SectionType.METHOD, 0.85),
    (["architecture"], SectionType.METHOD, 0.85),
    (["model"], SectionType.METHOD, 0.7),
    (["system"], SectionType.METHOD, 0.7),
    (["design"], SectionType.METHOD, 0.7),
    (["implementation"], SectionType.METHOD, 0.8),
    (["formulation"], SectionType.METHOD, 0.85),
    (["algorithm"], SectionType.METHOD, 0.8),
    (["technique"], SectionType.METHOD, 0.8),
    (["setup"], SectionType.METHOD, 0.7),
    (["training"], SectionType.METHOD, 0.7),
    # Results
    (["result"], SectionType.RESULTS, 1.0),
    (["experiment"], SectionType.RESULTS, 0.95),
    (["evaluation"], SectionType.RESULTS, 0.95),
    (["empirical"], SectionType.RESULTS, 0.9),
    (["benchmark"], SectionType.RESULTS, 0.85),
    (["performance"], SectionType.RESULTS, 0.8),
    (["comparison"], SectionType.RESULTS, 0.8),
    (["ablation"], SectionType.RESULTS, 0.9),
    (["analysis"], SectionType.RESULTS, 0.7),
    (["finding"], SectionType.RESULTS, 0.85),
    # Conclusion
    (["conclusion"], SectionType.CONCLUSION, 1.0),
    (["concluding"], SectionType.CONCLUSION, 0.95),
    (["summary"], SectionType.CONCLUSION, 0.8),
    (["future work"], SectionType.CONCLUSION, 0.85),
    (["limitation"], SectionType.CONCLUSION, 0.8),
    (["broader impact"], SectionType.CONCLUSION, 0.75),
    (["discussion"], SectionType.RESULTS, 0.6),  # Could be results or conclusion
]

# LLM classification prompt template
LLM_CLASSIFICATION_PROMPT = """You are classifying sections of academic computer science papers into standard categories.

Given the following section heading and the first 500 characters of its content, classify it into exactly ONE of these categories:
- abstract
- introduction  
- method
- results
- conclusion
- other

Respond with ONLY a JSON object: {{"section_type": "<category>", "confidence": <0.0-1.0>}}

Section heading: "{heading}"
Content preview: "{content_preview}"
"""


class SectionClassifier:
    """Classifies raw paper sections into standard categories.

    Uses a cascading approach: fast heuristics first, expensive LLM calls
    only when needed.

    Attributes:
        use_llm: Whether to use LLM for ambiguous sections.
        llm_client: Anthropic client instance (lazy initialized).
        llm_model: Model name for LLM classification.
        llm_calls_made: Counter for LLM API calls.
        max_llm_calls: Maximum LLM calls to make (cost control).
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "claude-sonnet-4-20250514",
        max_llm_calls: int = 10000,
    ):
        """Initialize the section classifier.

        Args:
            use_llm: Whether to enable LLM fallback classification.
            llm_model: Claude model to use for LLM classification.
            max_llm_calls: Maximum number of LLM API calls allowed.
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.max_llm_calls = max_llm_calls
        self.llm_calls_made = 0
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy initialization of the Anthropic client.

        Returns:
            Anthropic client instance.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        if self._llm_client is None:
            import anthropic
            self._llm_client = anthropic.Anthropic()
        return self._llm_client

    def classify_sections(
        self,
        raw_sections: list[RawSection],
        extraction_method: ExtractionMethod = ExtractionMethod.LATEX_PARSED,
    ) -> list[ClassifiedSection]:
        """Classify a list of raw sections into standard types.

        Args:
            raw_sections: List of RawSection objects to classify.
            extraction_method: How the sections were originally extracted.

        Returns:
            List of ClassifiedSection objects with assigned types.
        """
        classified = []

        for section in raw_sections:
            result = self._classify_one(section, extraction_method)
            classified.append(result)

        return classified

    def _classify_one(
        self,
        section: RawSection,
        extraction_method: ExtractionMethod,
    ) -> ClassifiedSection:
        """Classify a single section.

        Tries heuristic matching first, falls back to LLM if configured.

        Args:
            section: The raw section to classify.
            extraction_method: How the section was extracted.

        Returns:
            ClassifiedSection with type and confidence.
        """
        # Step 1: Try direct heuristic matching from model
        direct_type = SectionType.from_heading(section.heading)
        if direct_type != SectionType.OTHER:
            return ClassifiedSection(
                section_type=direct_type,
                original_heading=section.heading,
                content=section.content,
                confidence=1.0,
                method=extraction_method,
            )

        # Step 2: Try fuzzy heuristic matching
        fuzzy_result = self._fuzzy_match(section.heading)
        if fuzzy_result is not None:
            section_type, confidence = fuzzy_result
            return ClassifiedSection(
                section_type=section_type,
                original_heading=section.heading,
                content=section.content,
                confidence=confidence,
                method=extraction_method,
            )

        # Step 3: LLM fallback
        if self.use_llm and self.llm_calls_made < self.max_llm_calls:
            llm_result = self._classify_with_llm(section)
            if llm_result is not None:
                section_type, confidence = llm_result
                return ClassifiedSection(
                    section_type=section_type,
                    original_heading=section.heading,
                    content=section.content,
                    confidence=confidence,
                    method=ExtractionMethod.LLM_CLASSIFIED,
                )

        # Step 4: Default to OTHER
        return ClassifiedSection(
            section_type=SectionType.OTHER,
            original_heading=section.heading,
            content=section.content,
            confidence=0.0,
            method=extraction_method,
        )

    def _fuzzy_match(
        self, heading: str
    ) -> Optional[tuple[SectionType, float]]:
        """Attempt fuzzy keyword matching on section heading.

        Args:
            heading: Section heading text.

        Returns:
            Tuple of (SectionType, confidence) if matched, None otherwise.
        """
        heading_lower = heading.lower().strip()
        # Remove numbering prefixes like "3.", "III.", "3.1"
        heading_clean = heading_lower
        heading_clean = heading_clean.lstrip("0123456789ivx.")
        heading_clean = heading_clean.strip()

        best_match: Optional[tuple[SectionType, float]] = None
        best_confidence = 0.0

        for keywords, section_type, confidence in FUZZY_RULES:
            for keyword in keywords:
                if keyword in heading_clean:
                    if confidence > best_confidence:
                        best_match = (section_type, confidence)
                        best_confidence = confidence

        return best_match

    def _classify_with_llm(
        self, section: RawSection
    ) -> Optional[tuple[SectionType, float]]:
        """Use Claude to classify an ambiguous section.

        Args:
            section: The section to classify.

        Returns:
            Tuple of (SectionType, confidence), or None if LLM call fails.
        """
        try:
            content_preview = section.content[:500]
            prompt = LLM_CLASSIFICATION_PROMPT.format(
                heading=section.heading,
                content_preview=content_preview,
            )

            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )

            self.llm_calls_made += 1
            response_text = response.content[0].text.strip()

            # Parse JSON response
            result = json.loads(response_text)
            section_type_str = result.get("section_type", "other")
            confidence = float(result.get("confidence", 0.5))

            try:
                section_type = SectionType(section_type_str)
            except ValueError:
                section_type = SectionType.OTHER

            logger.debug(
                "LLM classified '%s' as %s (conf=%.2f)",
                section.heading,
                section_type.value,
                confidence,
            )
            return (section_type, confidence)

        except Exception as e:
            logger.warning(
                "LLM classification failed for '%s': %s", section.heading, e
            )
            return None

    def merge_duplicate_sections(
        self, sections: list[ClassifiedSection]
    ) -> list[ClassifiedSection]:
        """Merge multiple sections of the same type.

        Some papers split a single logical section (e.g., 'Method') across
        multiple LaTeX \\section{} commands. This merges them.

        Args:
            sections: List of classified sections.

        Returns:
            List with duplicate types merged (content concatenated).
        """
        merged: dict[SectionType, ClassifiedSection] = {}

        for section in sections:
            if section.section_type in merged:
                existing = merged[section.section_type]
                merged[section.section_type] = ClassifiedSection(
                    section_type=section.section_type,
                    original_heading=existing.original_heading,
                    content=existing.content + "\n\n" + section.content,
                    confidence=min(existing.confidence, section.confidence),
                    method=existing.method,
                )
            else:
                merged[section.section_type] = section

        return list(merged.values())
