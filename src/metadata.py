"""ArXiv metadata collection module.

This module handles fetching paper metadata from ArXiv using two strategies:
1. The ArXiv OAI-PMH API for bulk metadata harvesting.
2. The Kaggle ArXiv dataset as an alternative fast-start approach.

The OAI-PMH approach is preferred for completeness, while the Kaggle dataset
is useful for quick prototyping since it's a single JSON download.

Typical usage:
    ```python
    fetcher = MetadataFetcher(config)
    papers = fetcher.fetch_all_cs_papers("2022-01-01", "2025-12-31")
    ```
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import PaperMetadata

logger = logging.getLogger(__name__)

# ArXiv OAI-PMH endpoint
OAI_BASE_URL = "http://export.arxiv.org/oai2"
# ArXiv API endpoint
ARXIV_API_URL = "http://export.arxiv.org/api/query"
# OAI-PMH XML namespaces
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXivRaw/",
}


class MetadataFetcher:
    """Fetches and filters ArXiv paper metadata.

    Supports two data sources:
    - OAI-PMH bulk harvesting (complete, but slow due to rate limits)
    - Pre-downloaded Kaggle JSON dataset (fast, but may lag behind)

    Attributes:
        data_dir: Directory for storing metadata caches.
        request_delay: Seconds between API requests.
    """

    def __init__(self, data_dir: str = "./data", request_delay: float = 3.0):
        """Initialize the metadata fetcher.

        Args:
            data_dir: Base directory for data storage.
            request_delay: Delay between API requests in seconds.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "arxiv-section-extractor/0.1 (research project)"}
        )

    def fetch_via_oai(
        self,
        date_start: str,
        date_end: str,
        category_prefix: str = "cs",
    ) -> Generator[PaperMetadata, None, None]:
        """Fetch metadata using ArXiv OAI-PMH protocol.

        This harvests metadata in bulk using the OAI-PMH ListRecords verb.
        Results are paginated via resumption tokens. Each page returns ~1000
        records. Rate limiting is enforced server-side (typically 3s between
        requests).

        Args:
            date_start: Start date in 'YYYY-MM-DD' format.
            date_end: End date in 'YYYY-MM-DD' format.
            category_prefix: ArXiv category prefix to filter (default: 'cs').

        Yields:
            PaperMetadata for each matching paper.
        """
        logger.info(
            "Starting OAI-PMH harvest for %s papers from %s to %s",
            category_prefix,
            date_start,
            date_end,
        )

        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXivRaw",
            "from": date_start,
            "until": date_end,
            "set": f"{category_prefix}",
        }

        total_fetched = 0
        resumption_token = None

        while True:
            if resumption_token:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}

            response = self._make_oai_request(params)
            if response is None:
                logger.error("OAI-PMH request failed, stopping harvest.")
                break

            root = ET.fromstring(response.text)

            # Extract records
            records = root.findall(".//oai:record", OAI_NS)
            for record in records:
                metadata = self._parse_oai_record(record)
                if metadata and self._is_cs_paper(metadata, category_prefix):
                    total_fetched += 1
                    yield metadata

            # Check for resumption token
            token_elem = root.find(".//oai:resumptionToken", OAI_NS)
            if token_elem is not None and token_elem.text:
                resumption_token = token_elem.text.strip()
                logger.info(
                    "Fetched %d papers so far, continuing with resumption token...",
                    total_fetched,
                )
            else:
                logger.info("Harvest complete. Total papers: %d", total_fetched)
                break

            time.sleep(self.request_delay)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=60))
    def _make_oai_request(self, params: dict) -> Optional[requests.Response]:
        """Make a request to the OAI-PMH endpoint with retries.

        Args:
            params: Query parameters for the OAI request.

        Returns:
            The HTTP response, or None if all retries failed.
        """
        response = self._session.get(OAI_BASE_URL, params=params, timeout=60)
        response.raise_for_status()

        # Check for OAI-PMH errors (e.g., 503 retry-after)
        if response.status_code == 503:
            retry_after = int(response.headers.get("Retry-After", 30))
            logger.warning("OAI-PMH 503, retrying after %d seconds", retry_after)
            time.sleep(retry_after)
            raise requests.exceptions.RetryError("503 retry-after")

        return response

    def _parse_oai_record(self, record: ET.Element) -> Optional[PaperMetadata]:
        """Parse a single OAI-PMH record into PaperMetadata.

        Args:
            record: XML element representing one OAI record.

        Returns:
            Parsed PaperMetadata, or None if parsing fails.
        """
        try:
            metadata = record.find(".//arxiv:arXivRaw", OAI_NS)
            if metadata is None:
                return None

            arxiv_id = self._get_text(metadata, "arxiv:id", OAI_NS)
            if not arxiv_id:
                return None

            categories_str = self._get_text(metadata, "arxiv:categories", OAI_NS) or ""
            categories = categories_str.split()

            return PaperMetadata(
                arxiv_id=arxiv_id,
                title=self._get_text(metadata, "arxiv:title", OAI_NS) or "",
                authors=self._parse_authors(
                    self._get_text(metadata, "arxiv:authors", OAI_NS) or ""
                ),
                categories=categories,
                primary_category=categories[0] if categories else "",
                published=self._get_text(metadata, "arxiv:datestamp", OAI_NS) or "",
                updated=self._get_text(metadata, "arxiv:datestamp", OAI_NS) or "",
                abstract=self._get_text(metadata, "arxiv:abstract", OAI_NS) or "",
                doi=self._get_text(metadata, "arxiv:doi", OAI_NS),
            )
        except Exception as e:
            logger.warning("Failed to parse OAI record: %s", e)
            return None

    def load_from_kaggle_json(
        self,
        json_path: str,
        date_start: str = "2022-01-01",
        date_end: str = "2025-12-31",
        category_prefix: str = "cs.",
    ) -> Generator[PaperMetadata, None, None]:
        """Load metadata from the Kaggle ArXiv dataset JSON file.

        The Kaggle dataset (https://www.kaggle.com/datasets/Cornell-University/arxiv)
        provides a single JSON-lines file with all ArXiv metadata. This is much
        faster than OAI-PMH for initial setup.

        Args:
            json_path: Path to the arxiv-metadata-oai-snapshot.json file.
            date_start: Start date filter (inclusive).
            date_end: End date filter (inclusive).
            category_prefix: Category prefix to filter by.

        Yields:
            PaperMetadata for each matching paper.
        """
        logger.info("Loading metadata from Kaggle JSON: %s", json_path)
        start_dt = datetime.fromisoformat(date_start)
        end_dt = datetime.fromisoformat(date_end)
        count = 0

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by category
                categories = entry.get("categories", "").split()
                if not any(cat.startswith(category_prefix) for cat in categories):
                    continue

                # Filter by date
                versions = entry.get("versions", [])
                if not versions:
                    continue

                first_version_date = versions[0].get("created", "")
                try:
                    paper_date = datetime.strptime(
                        first_version_date, "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except ValueError:
                    continue

                if not (start_dt <= paper_date <= end_dt):
                    continue

                count += 1
                yield PaperMetadata(
                    arxiv_id=entry.get("id", ""),
                    title=entry.get("title", "").replace("\n", " ").strip(),
                    authors=self._parse_kaggle_authors(
                        entry.get("authors_parsed", [])
                    ),
                    categories=categories,
                    primary_category=categories[0] if categories else "",
                    published=paper_date.isoformat(),
                    updated=entry.get("update_date", ""),
                    abstract=entry.get("abstract", "").strip(),
                    doi=entry.get("doi"),
                )

        logger.info("Loaded %d papers from Kaggle JSON", count)

    def save_metadata_cache(
        self, papers: list[PaperMetadata], cache_path: Optional[str] = None
    ) -> Path:
        """Save fetched metadata to a local JSON-lines cache file.

        Args:
            papers: List of paper metadata to cache.
            cache_path: Optional custom path. Defaults to data_dir/metadata_cache.jsonl.

        Returns:
            Path to the saved cache file.
        """
        path = Path(cache_path) if cache_path else self.data_dir / "metadata_cache.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper.__dict__) + "\n")

        logger.info("Saved %d paper metadata entries to %s", len(papers), path)
        return path

    def load_metadata_cache(
        self, cache_path: Optional[str] = None
    ) -> list[PaperMetadata]:
        """Load metadata from a previously saved cache file.

        Args:
            cache_path: Path to the cache file. Defaults to data_dir/metadata_cache.jsonl.

        Returns:
            List of PaperMetadata loaded from cache.
        """
        path = Path(cache_path) if cache_path else self.data_dir / "metadata_cache.jsonl"
        papers = []

        if not path.exists():
            logger.warning("No metadata cache found at %s", path)
            return papers

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    papers.append(PaperMetadata(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed cache entry: %s", e)

        logger.info("Loaded %d papers from cache %s", len(papers), path)
        return papers

    @staticmethod
    def _get_text(
        element: ET.Element, tag: str, ns: dict
    ) -> Optional[str]:
        """Safely extract text from an XML element.

        Args:
            element: Parent XML element.
            tag: Tag name to find.
            ns: Namespace dictionary.

        Returns:
            Text content of the element, or None.
        """
        child = element.find(tag, ns)
        return child.text.strip() if child is not None and child.text else None

    @staticmethod
    def _parse_authors(authors_str: str) -> list[str]:
        """Parse author string from OAI format.

        Args:
            authors_str: Raw author string (newline or comma separated).

        Returns:
            List of author name strings.
        """
        if not authors_str:
            return []
        # OAI format has authors separated by newlines with affiliation info
        authors = []
        for line in authors_str.split("\n"):
            name = line.strip().split("(")[0].strip()  # Remove affiliations
            if name:
                authors.append(name)
        return authors

    @staticmethod
    def _parse_kaggle_authors(authors_parsed: list) -> list[str]:
        """Parse authors from Kaggle dataset format.

        Args:
            authors_parsed: List of [last, first, suffix] lists.

        Returns:
            List of 'First Last' formatted author names.
        """
        authors = []
        for parts in authors_parsed:
            if len(parts) >= 2:
                first = parts[1].strip()
                last = parts[0].strip()
                name = f"{first} {last}".strip()
                if name:
                    authors.append(name)
        return authors

    @staticmethod
    def _is_cs_paper(paper: PaperMetadata, prefix: str = "cs") -> bool:
        """Check if a paper belongs to the target category.

        Args:
            paper: Paper metadata to check.
            prefix: Category prefix to match.

        Returns:
            True if any of the paper's categories match the prefix.
        """
        return any(cat.startswith(f"{prefix}.") for cat in paper.categories)
