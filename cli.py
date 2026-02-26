"""Command-line interface for the ArXiv section extractor.

Provides commands for running the full pipeline or individual steps.

Usage:
    # Full pipeline
    python -m src.cli run --config configs/default.yaml

    # Metadata only
    python -m src.cli metadata --source oai --output data/metadata.jsonl

    # Process a single paper (for testing)
    python -m src.cli single --arxiv-id 2301.12345

    # Resume from checkpoint
    python -m src.cli run --config configs/default.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("arxiv_extractor.log"),
        ],
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the full pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    from src.metadata import MetadataFetcher
    from src.pipeline import Pipeline

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_logging(config.get("processing", {}).get("log_level", "INFO"))
    logger = logging.getLogger(__name__)

    # Step 1: Load metadata
    logger.info("Step 1: Loading paper metadata...")
    fetcher = MetadataFetcher(
        data_dir=config.get("download", {}).get("data_dir", "./data")
    )

    query_config = config.get("query", {})
    date_start = query_config.get("date_start", "2022-01-01")
    date_end = query_config.get("date_end", "2025-12-31")

    # Try to load from cache first
    metadata_list = fetcher.load_metadata_cache()

    if not metadata_list:
        if args.kaggle_json:
            logger.info("Loading metadata from Kaggle JSON...")
            metadata_list = list(
                fetcher.load_from_kaggle_json(
                    args.kaggle_json,
                    date_start=date_start,
                    date_end=date_end,
                )
            )
        else:
            logger.info("Fetching metadata via OAI-PMH (this may take a while)...")
            metadata_list = list(
                fetcher.fetch_via_oai(
                    date_start=date_start,
                    date_end=date_end,
                )
            )
        # Cache for future runs
        fetcher.save_metadata_cache(metadata_list)

    logger.info("Loaded %d papers", len(metadata_list))

    # Step 2: Run pipeline
    pipeline = Pipeline(config)
    output_path = pipeline.run(metadata_list)

    logger.info("Done! Output written to: %s", output_path)


def cmd_single(args: argparse.Namespace) -> None:
    """Process a single paper (for testing and debugging).

    Args:
        args: Parsed CLI arguments with arxiv_id.
    """
    setup_logging("DEBUG")
    logger = logging.getLogger(__name__)

    from src.downloader import Downloader
    from src.latex_parser import LatexParser
    from src.models import ExtractionMethod
    from src.section_classifier import SectionClassifier

    downloader = Downloader()
    parser = LatexParser()
    classifier = SectionClassifier(use_llm=args.use_llm)

    arxiv_id = args.arxiv_id
    logger.info("Processing paper: %s", arxiv_id)

    # Check source availability
    if not downloader.has_latex_source(arxiv_id):
        logger.error("No LaTeX source available for %s (PDF-only). Skipping.", arxiv_id)
        return

    # Download source
    source_dir = downloader.download_source(arxiv_id)
    if not source_dir:
        logger.error("Failed to download source for %s", arxiv_id)
        return

    logger.info("LaTeX source found, parsing...")
    raw_sections = parser.parse_source_directory(source_dir)

    if not raw_sections:
        logger.error("Could not extract any sections from %s", arxiv_id)
        return

    logger.info("Extracted %d raw sections:", len(raw_sections))
    for s in raw_sections:
        logger.info("  [%d] %s (%d chars)", s.order, s.heading, len(s.content))

    # Classify
    classified = classifier.classify_sections(raw_sections, ExtractionMethod.LATEX_PARSED)
    classified = classifier.merge_duplicate_sections(classified)

    logger.info("\nClassified sections:")
    for s in classified:
        logger.info(
            "  %s (was: '%s', conf=%.2f, method=%s) — %d chars",
            s.section_type.value,
            s.original_heading,
            s.confidence,
            s.method.value,
            len(s.content),
        )


def cmd_convert(args: argparse.Namespace) -> None:
    """Batch convert LaTeX sources to plain text .txt files.

    Args:
        args: Parsed CLI arguments with sources_dir and output_dir.
    """
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    from src.tex_to_txt import batch_convert_tex_to_txt

    sources_dir = Path(args.sources_dir)
    output_dir = Path(args.output_dir)

    logger.info("Converting .tex → .txt: %s → %s", sources_dir, output_dir)
    results = batch_convert_tex_to_txt(sources_dir, output_dir)
    logger.info(
        "Done: %d succeeded, %d failed, %d skipped",
        results["succeeded"],
        results["failed"],
        results["skipped"],
    )


def cmd_metadata(args: argparse.Namespace) -> None:
    """Fetch and save metadata only.

    Args:
        args: Parsed CLI arguments.
    """
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    from src.metadata import MetadataFetcher

    fetcher = MetadataFetcher(data_dir=args.data_dir)

    if args.source == "kaggle" and args.kaggle_json:
        papers = list(
            fetcher.load_from_kaggle_json(
                args.kaggle_json,
                date_start=args.date_start,
                date_end=args.date_end,
            )
        )
    else:
        papers = list(
            fetcher.fetch_via_oai(
                date_start=args.date_start,
                date_end=args.date_end,
            )
        )

    output_path = fetcher.save_metadata_cache(papers, args.output)
    logger.info("Saved %d papers to %s", len(papers), output_path)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="ArXiv Section Extractor — Extract structured sections from CS papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    run_parser.add_argument(
        "--kaggle-json", default=None, help="Path to Kaggle ArXiv JSON for metadata"
    )

    # Single paper command
    single_parser = subparsers.add_parser("single", help="Process a single paper")
    single_parser.add_argument("arxiv_id", help="ArXiv paper ID (e.g., 2301.12345)")
    single_parser.add_argument(
        "--use-llm", action="store_true", help="Use LLM for ambiguous sections"
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Batch convert LaTeX sources to plain text"
    )
    convert_parser.add_argument(
        "--sources-dir", default="./data/sources", help="Directory with extracted LaTeX sources"
    )
    convert_parser.add_argument(
        "--output-dir", default="./data/txt", help="Directory for output .txt files"
    )

    # Metadata command
    meta_parser = subparsers.add_parser("metadata", help="Fetch metadata only")
    meta_parser.add_argument(
        "--source", choices=["oai", "kaggle"], default="oai", help="Metadata source"
    )
    meta_parser.add_argument("--kaggle-json", default=None, help="Kaggle JSON path")
    meta_parser.add_argument("--data-dir", default="./data", help="Data directory")
    meta_parser.add_argument("--output", default=None, help="Output file path")
    meta_parser.add_argument("--date-start", default="2022-01-01", help="Start date")
    meta_parser.add_argument("--date-end", default="2025-12-31", help="End date")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "metadata":
        cmd_metadata(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
