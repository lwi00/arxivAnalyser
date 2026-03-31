"""Pipeline orchestrator for the ArXiv section extraction workflow.

This module ties together all components (metadata, download, parse, classify,
output) into a complete end-to-end pipeline. It supports:

- Checkpoint-based resumption for long-running jobs
- Parallel processing with configurable worker count
- Progress tracking and logging
- Incremental Parquet writes to avoid memory issues

The pipeline processes papers in batches:
1. Load or fetch metadata for target papers.
2. Download LaTeX sources (with PDF fallback).
3. Parse and classify sections.
4. Write results to Parquet in chunks.

Typical usage:
    ```python
    pipeline = Pipeline.from_config("configs/default.yaml")
    pipeline.run()
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from src.downloader import Downloader
from src.latex_parser import LatexParser
from src.models import (
    ClassifiedSection,
    ExtractionMethod,
    PaperMetadata,
    PaperRecord,
    ProcessingStatus,
    SectionType,
)
from src.section_classifier import SectionClassifier
from src.tex_to_txt import _sanitize_filename

logger = logging.getLogger(__name__)


def _process_single_paper_worker(args: tuple) -> Optional[dict]:
    """Worker function for parallel processing (must be top-level for pickling).

    This processes a single paper through LaTeX extraction only:
    source parsing → section classification → record creation.

    Args:
        args: Tuple of (source_dir_str, metadata_dict, config_dict)

    Returns:
        Dictionary representation of PaperRecord, or None if processing failed.
    """
    source_dir_str, metadata_dict, config_dict = args

    parser = LatexParser(
        max_file_size=config_dict.get("latex", {}).get("max_file_size", 10_000_000)
    )
    classifier = SectionClassifier(
        use_llm=False  # No LLM in worker processes (use main process for that)
    )

    metadata = PaperMetadata(**metadata_dict)
    extraction_method = ExtractionMethod.LATEX_PARSED

    raw_sections = parser.parse_source_directory(source_dir_str)
    if not raw_sections:
        return None

    # Classify sections
    classified = classifier.classify_sections(raw_sections, extraction_method)
    classified = classifier.merge_duplicate_sections(classified)

    # Build record
    record = _build_paper_record(metadata, classified, extraction_method, len(raw_sections))
    result = asdict(record)
    # Attach classified sections for txt export (avoids re-parsing)
    result["_sections"] = [
        {
            "heading": s.original_heading,
            "content": s.content,
            "section_type": s.section_type.value,
            "order": i,
        }
        for i, s in enumerate(classified)
    ]
    return result


def _build_paper_record(
    metadata: PaperMetadata,
    sections: list[ClassifiedSection],
    extraction_method: ExtractionMethod,
    raw_count: int,
) -> PaperRecord:
    """Build a PaperRecord from metadata and classified sections.

    Args:
        metadata: Paper metadata.
        sections: List of classified sections.
        extraction_method: How sections were extracted.
        raw_count: Number of raw sections before merging.

    Returns:
        Populated PaperRecord ready for output.
    """
    section_map = {s.section_type: s for s in sections}
    found_types = [s.section_type.value for s in sections if s.section_type != SectionType.OTHER]

    return PaperRecord(
        arxiv_id=metadata.arxiv_id,
        title=metadata.title,
        authors="; ".join(metadata.authors),
        categories=", ".join(metadata.categories),
        primary_category=metadata.primary_category,
        published=metadata.published,
        updated=metadata.updated,
        abstract=metadata.abstract,
        related_work=section_map.get(SectionType.RELATED_WORK, _empty_section()).content,
        extraction_method=extraction_method.value,
        extraction_success=SectionType.RELATED_WORK in section_map,
        sections_found=", ".join(found_types),
        raw_section_count=raw_count,
    )


def _empty_section() -> ClassifiedSection:
    """Create an empty placeholder section.

    Returns:
        ClassifiedSection with empty content.
    """
    return ClassifiedSection(
        section_type=SectionType.OTHER,
        original_heading="",
        content="",
    )


class Pipeline:
    """Orchestrates the full paper processing pipeline.

    Manages the end-to-end flow from metadata collection through to
    Parquet output, with support for checkpointing and parallel execution.

    Attributes:
        config: Full configuration dictionary.
        downloader: Paper source/PDF downloader.
        parser: LaTeX source parser.
        pdf_extractor: PDF text extractor.
        classifier: Section classifier.
        progress: Dict tracking which papers have been processed.
    """

    def __init__(self, config: dict):
        """Initialize the pipeline from a config dictionary.

        Args:
            config: Full configuration dict (typically loaded from YAML).
        """
        self.config = config

        dl_config = config.get("download", {})
        self.downloader = Downloader(
            sources_dir=dl_config.get("sources_dir", "./data/sources"),
            pdfs_dir=dl_config.get("pdfs_dir", "./data/pdfs"),
            max_concurrent=dl_config.get("max_concurrent", 5),
            request_delay=dl_config.get("request_delay", 3.0),
            max_retries=dl_config.get("max_retries", 3),
        )

        latex_config = config.get("latex", {})
        self.parser = LatexParser(
            max_file_size=latex_config.get("max_file_size", 10_000_000)
        )

        cls_config = config.get("classification", {})
        llm_config = cls_config.get("llm", {})
        self.classifier = SectionClassifier(
            use_llm=cls_config.get("use_llm_fallback", True),
            llm_model=llm_config.get("model", "claude-sonnet-4-20250514"),
            max_llm_calls=llm_config.get("max_llm_calls", 10000),
        )

        proc_config = config.get("processing", {})
        self.num_workers = proc_config.get("num_workers", 4)
        self.resume = proc_config.get("resume", True)
        self.max_papers_per_run = proc_config.get("max_papers_per_run", 0)
        self.progress_file = Path(proc_config.get("progress_file", "./data/progress.json"))

        out_config = config.get("output", {})
        self.output_path = Path(out_config.get("path", "./output/arxiv_cs_related_works.parquet"))
        self.compression = out_config.get("compression", "snappy")
        self.checkpoint_interval = out_config.get("checkpoint_interval", 1000)
        self.txt_export = out_config.get("txt_export", False)
        self.txt_dir = Path(out_config.get("txt_dir", "./data/txt"))
        self.txt_related_only = out_config.get("txt_related_only", False)
        self.txt_only_mode = out_config.get("txt_only_mode", False)

        self.filter_date_start: Optional[str] = proc_config.get("filter_date_start")
        self.filter_date_end: Optional[str] = proc_config.get("filter_date_end")

        self.progress: dict[str, str] = {}
        if self.resume:
            self._load_progress()

    @classmethod
    def from_config(cls, config_path: str) -> Pipeline:
        """Create a Pipeline from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configured Pipeline instance.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(self, metadata_list: list[PaperMetadata]) -> Path:
        """Execute the full pipeline (LaTeX-only mode).

        Steps:
        1. Filter papers that have LaTeX source available (HEAD probing).
        2. Download LaTeX sources for papers that pass the filter.
        3. Parse and classify sections.
        4. Write results to Parquet.

        Args:
            metadata_list: List of paper metadata to process.

        Returns:
            Path to the output Parquet file.
        """
        logger.info("Starting pipeline for %d papers", len(metadata_list))
        start_time = time.time()

        # Filter out all previously attempted papers (completed, failed, or no_latex_source)
        if self.resume:
            pending = [
                m for m in metadata_list
                if m.arxiv_id not in self.progress
            ]
            logger.info(
                "Resuming: %d already attempted, %d new",
                len(metadata_list) - len(pending),
                len(pending),
            )
        else:
            pending = metadata_list

        # Apply date filter (if configured)
        if self.filter_date_start or self.filter_date_end:
            pending = self._filter_by_date(pending)

        # Apply per-run paper limit
        if self.max_papers_per_run > 0 and len(pending) > self.max_papers_per_run:
            logger.info(
                "Limiting to %d papers for this run (out of %d pending)",
                self.max_papers_per_run,
                len(pending),
            )
            pending = pending[: self.max_papers_per_run]

        # Step 1: Filter papers with LaTeX source available
        if self.config.get("download", {}).get("probe_source_availability", True):
            pending = self._filter_latex_available(pending)

        # Process in batches
        batch_size = self.checkpoint_interval
        total_batches = (len(pending) + batch_size - 1) // batch_size

        if self.txt_only_mode:
            # Fast path: only produce .txt files, skip Parquet entirely
            self.txt_export = True
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(pending))
                batch = pending[batch_start:batch_end]
                logger.info(
                    "Processing batch %d/%d (%d papers) [txt-only mode]",
                    batch_idx + 1, total_batches, len(batch),
                )
                self._process_batch(batch)

            elapsed = time.time() - start_time
            logger.info(
                "Pipeline complete (txt-only). %d papers processed in %.1f seconds. Output: %s",
                len(pending), elapsed, self.txt_dir,
            )
            return self.txt_dir
        else:
            # Full pipeline: Parquet output with optional txt export
            existing_records = self._load_existing_records()
            new_records: list[dict] = []

            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(pending))
                batch = pending[batch_start:batch_end]

                logger.info(
                    "Processing batch %d/%d (%d papers)",
                    batch_idx + 1, total_batches, len(batch),
                )

                batch_records = self._process_batch(batch)
                new_records.extend(batch_records)

                # Write checkpoint (existing + new)
                self._write_checkpoint(existing_records + new_records)
                logger.info(
                    "Checkpoint saved. %d new records this run, %d total",
                    len(new_records),
                    len(existing_records) + len(new_records),
                )

            # Final output: merge existing + new, deduplicate by arxiv_id
            all_records = self._merge_records(existing_records, new_records)
            output_path = self._write_parquet(all_records)

            elapsed = time.time() - start_time
            logger.info(
                "Pipeline complete. %d new + %d existing = %d total records in %.1f seconds",
                len(new_records), len(existing_records), len(all_records), elapsed,
            )
            return output_path

    def _filter_by_date(
        self, papers: list[PaperMetadata]
    ) -> list[PaperMetadata]:
        """Filter papers by publication date range.

        Uses the ``published`` field (ISO format string) for comparison.
        Only papers whose publication date falls within the configured
        ``filter_date_start`` / ``filter_date_end`` range are kept.

        Args:
            papers: List of paper metadata to filter.

        Returns:
            Filtered list containing only papers within the date range.
        """
        before = len(papers)
        filtered = papers
        if self.filter_date_start:
            filtered = [p for p in filtered if p.published[:10] >= self.filter_date_start]
        if self.filter_date_end:
            filtered = [p for p in filtered if p.published[:10] <= self.filter_date_end]

        logger.info(
            "Date filter [%s .. %s]: %d → %d papers (%d filtered out)",
            self.filter_date_start or "*",
            self.filter_date_end or "*",
            before,
            len(filtered),
            before - len(filtered),
        )
        return filtered

    def _filter_latex_available(
        self, papers: list[PaperMetadata]
    ) -> list[PaperMetadata]:
        """Filter papers to only those with LaTeX source available.

        Uses async HEAD requests to probe ArXiv's e-print endpoint.
        Papers that return a source content type (tar.gz) are kept;
        those that return PDF or 404 are skipped.

        This avoids downloading ~10-15% of papers that only have PDFs,
        saving significant time and bandwidth.

        Args:
            papers: Full list of paper metadata.

        Returns:
            Filtered list containing only papers with LaTeX source.
        """
        logger.info(
            "Probing source availability for %d papers (HEAD requests)...",
            len(papers),
        )

        # Skip papers we've already marked as no-source
        to_check = [
            p for p in papers
            if self.progress.get(p.arxiv_id) != "no_latex_source"
        ]

        # Process in chunks to avoid overwhelming the event loop
        chunk_size = 500
        available_ids: set[str] = set()
        checked = 0

        for i in range(0, len(to_check), chunk_size):
            chunk = to_check[i : i + chunk_size]
            chunk_ids = [p.arxiv_id for p in chunk]

            results = asyncio.run(
                self.downloader.batch_check_latex_source(chunk_ids)
            )

            for arxiv_id, has_source in results.items():
                if has_source:
                    available_ids.add(arxiv_id)
                else:
                    self.progress[arxiv_id] = "no_latex_source"

            checked += len(chunk)
            logger.info(
                "Probed %d/%d papers — %d with LaTeX source so far",
                checked,
                len(to_check),
                len(available_ids),
            )

        self._save_progress()

        filtered = [p for p in papers if p.arxiv_id in available_ids]
        skipped = len(papers) - len(filtered)
        logger.info(
            "Source availability filter: %d papers with LaTeX, %d skipped (PDF-only or missing)",
            len(filtered),
            skipped,
        )
        return filtered

    def _process_batch(self, batch: list[PaperMetadata]) -> list[dict]:
        """Process a batch of papers (LaTeX source only).

        Downloads LaTeX sources, then parses papers in parallel.

        Args:
            batch: List of paper metadata for this batch.

        Returns:
            List of PaperRecord dicts for successfully processed papers.
        """
        records = []

        # Step 1: Download LaTeX sources (async batch for concurrency)
        arxiv_ids = [m.arxiv_id for m in batch]
        logger.info("Downloading %d sources (async, concurrency=%d)...", len(arxiv_ids), self.downloader.max_concurrent)
        source_paths: dict[str, Optional[Path]] = asyncio.run(
            self.downloader.batch_download_sources(arxiv_ids)
        )
        for aid, path in source_paths.items():
            if path is None:
                self.progress[aid] = ProcessingStatus.FAILED_DOWNLOAD.value

        # Step 2: Process papers in parallel
        worker_args = []
        for metadata in batch:
            src = source_paths.get(metadata.arxiv_id)
            if src is None:
                continue

            worker_args.append((
                str(src),
                metadata.__dict__,
                self.config,
            ))

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(_process_single_paper_worker, args): args[1]["arxiv_id"]
                for args in worker_args
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing",
                unit="paper",
            ):
                arxiv_id = futures[future]
                try:
                    result = future.result(timeout=120)
                    if result:
                        records.append(result)
                        self.progress[arxiv_id] = ProcessingStatus.COMPLETED.value
                    else:
                        self.progress[arxiv_id] = ProcessingStatus.FAILED_PARSE.value
                except Exception as e:
                    logger.warning("Worker failed for %s: %s", arxiv_id, e)
                    self.progress[arxiv_id] = ProcessingStatus.FAILED_PARSE.value

        # Step 3: Optional .txt export (reuses sections from workers, no re-parsing)
        if self.txt_export:
            for record in records:
                sections = record.pop("_sections", [])
                if sections:
                    try:
                        self._write_txt_sections(record["arxiv_id"], sections)
                    except Exception as e:
                        logger.warning("txt export failed for %s: %s", record["arxiv_id"], e)

        self._save_progress()
        return records

    def _write_parquet(self, records: list[dict]) -> Path:
        """Write records to a Parquet file.

        Args:
            records: List of PaperRecord dicts.

        Returns:
            Path to the written Parquet file.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            logger.warning("No records to write!")
            return self.output_path

        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(self.output_path), compression=self.compression)

        logger.info("Wrote %d records to %s", len(records), self.output_path)
        return self.output_path

    def _write_checkpoint(self, records: list[dict]) -> None:
        """Write intermediate Parquet checkpoint.

        Args:
            records: All records collected so far.
        """
        checkpoint_path = self.output_path.with_suffix(".checkpoint.parquet")
        if records:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(checkpoint_path), compression=self.compression)

    def _load_existing_records(self) -> list[dict]:
        """Load records from a previous run's Parquet file.

        Returns:
            List of record dicts from the existing file, or empty list if
            no file exists.
        """
        if not self.output_path.exists():
            return []

        try:
            df = pd.read_parquet(self.output_path)
            records = df.to_dict("records")
            logger.info(
                "Loaded %d existing records from %s",
                len(records),
                self.output_path,
            )
            return records
        except Exception as e:
            logger.warning("Failed to load existing Parquet: %s", e)
            return []

    @staticmethod
    def _merge_records(
        existing: list[dict], new: list[dict]
    ) -> list[dict]:
        """Merge existing and new records, deduplicating by arxiv_id.

        New records take precedence over existing ones if there are duplicates.

        Args:
            existing: Records from previous runs.
            new: Records from the current run.

        Returns:
            Combined list with no duplicate arxiv_ids.
        """
        new_ids = {r["arxiv_id"] for r in new}
        deduplicated_existing = [r for r in existing if r["arxiv_id"] not in new_ids]
        return deduplicated_existing + new

    def _load_progress(self) -> None:
        """Load progress from the progress tracking file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    self.progress = json.load(f)
                logger.info("Loaded progress: %d entries", len(self.progress))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load progress file: %s", e)
                self.progress = {}

    def _save_progress(self) -> None:
        """Save progress to the tracking file."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f)

    def _write_txt_sections(self, arxiv_id: str, sections: list[dict]) -> None:
        """Write classified sections as .txt files for a single paper.

        Uses sections already parsed by workers — no re-parsing needed.

        Args:
            arxiv_id: ArXiv identifier for the paper.
            sections: List of section dicts with heading, content, section_type, order.
        """
        if self.txt_related_only:
            sections = [s for s in sections if s.get("section_type") == "related_work"]
        if not sections:
            return

        paper_dir = self.txt_dir / arxiv_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        for section in sections:
            filename = f"{section['order']:02d}_{_sanitize_filename(section['heading'])}.txt"
            (paper_dir / filename).write_text(section["content"], encoding="utf-8")
