"""ArXiv source and PDF downloader.

This module handles downloading paper source files (LaTeX .tar.gz) and PDFs
from ArXiv. It supports two download strategies:

1. **Direct HTTP** - Downloads from `https://arxiv.org/e-print/{id}` one at a time.
   Suitable for smaller batches (<10k papers). Rate-limited to respect ArXiv's
   servers (3s delay between requests).

2. **S3 Bulk Access** - Uses the ArXiv requester-pays S3 bucket (`s3://arxiv/`)
   for large-scale downloads. Requires AWS credentials. Much faster and the
   recommended approach for 100k+ papers.

Typical usage:
    ```python
    downloader = Downloader(config)
    source_path = downloader.download_source("2301.12345")
    if source_path is None:
        pdf_path = downloader.download_pdf("2301.12345")
    ```
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import tarfile
import time
from pathlib import Path
from typing import Optional

import aiohttp
import certifi
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ArXiv endpoints
EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"


class Downloader:
    """Downloads ArXiv paper sources and PDFs.

    Handles both single downloads and async batch downloads with
    rate limiting and retry logic.

    Attributes:
        sources_dir: Directory for storing LaTeX source archives.
        pdfs_dir: Directory for storing fallback PDFs.
        max_concurrent: Maximum concurrent downloads.
        request_delay: Delay between sequential requests.
    """

    # Content types that indicate LaTeX source availability
    SOURCE_CONTENT_TYPES = {"application/x-eprint-tar", "application/gzip", "application/x-gzip"}
    # Content types that indicate PDF-only (no LaTeX source)
    PDF_CONTENT_TYPES = {"application/pdf"}

    def __init__(
        self,
        sources_dir: str = "./data/sources",
        pdfs_dir: str = "./data/pdfs",
        max_concurrent: int = 5,
        request_delay: float = 3.0,
        max_retries: int = 3,
    ):
        """Initialize the downloader.

        Args:
            sources_dir: Directory for LaTeX source files.
            pdfs_dir: Directory for fallback PDF files.
            max_concurrent: Max concurrent async downloads.
            request_delay: Seconds between requests (rate limiting).
            max_retries: Number of retry attempts per download.
        """
        self.sources_dir = Path(sources_dir)
        self.pdfs_dir = Path(pdfs_dir)
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.max_retries = max_retries

        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "arxiv-section-extractor/0.1 (research project)"}
        )

    def has_latex_source(self, arxiv_id: str) -> bool:
        """Check if a paper has LaTeX source available using a HEAD request.

        This is a lightweight probe that avoids downloading the full archive.
        ArXiv returns different Content-Type headers depending on availability:
        - ``application/x-eprint-tar`` or ``application/gzip`` → source exists
        - ``application/pdf`` → only PDF available (no source)
        - 404 → paper not found

        Args:
            arxiv_id: ArXiv paper identifier (e.g., '2301.12345').

        Returns:
            True if LaTeX source is available, False otherwise.
        """
        url = EPRINT_URL.format(arxiv_id=arxiv_id)
        try:
            time.sleep(self.request_delay)
            response = self._session.head(url, timeout=30, allow_redirects=True)
            if response.status_code == 404:
                return False
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            return content_type not in self.PDF_CONTENT_TYPES
        except Exception as e:
            logger.debug("HEAD request failed for %s: %s", arxiv_id, e)
            return False

    async def batch_check_latex_source(
        self, arxiv_ids: list[str], progress_callback=None
    ) -> dict[str, bool]:
        """Check source availability for multiple papers concurrently.

        Uses async HEAD requests to quickly probe which papers have
        LaTeX source, before committing to full downloads.

        Args:
            arxiv_ids: List of ArXiv IDs to check.
            progress_callback: Optional callable(arxiv_id, has_source) for tracking.

        Returns:
            Dictionary mapping arxiv_id to availability boolean.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "arxiv-section-extractor/0.1 (research project)"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            tasks = [
                self._async_check_one(session, semaphore, arxiv_id, progress_callback)
                for arxiv_id in arxiv_ids
            ]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for arxiv_id, result in zip(arxiv_ids, completed):
                if isinstance(result, Exception):
                    results[arxiv_id] = False
                else:
                    results[arxiv_id] = result

        return results

    async def _async_check_one(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        arxiv_id: str,
        progress_callback=None,
    ) -> bool:
        """Check source availability for a single paper asynchronously.

        Args:
            session: aiohttp session.
            semaphore: Concurrency limiter.
            arxiv_id: Paper ID to check.
            progress_callback: Optional progress reporter.

        Returns:
            True if LaTeX source is available.
        """
        async with semaphore:
            url = EPRINT_URL.format(arxiv_id=arxiv_id)
            try:
                async with session.head(url, allow_redirects=True) as response:
                    if response.status == 404:
                        has_source = False
                    else:
                        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                        has_source = content_type not in self.PDF_CONTENT_TYPES

                    if progress_callback:
                        progress_callback(arxiv_id, has_source)
                    return has_source
            except Exception as e:
                logger.debug("Probe failed for %s: %s", arxiv_id, e)
                if progress_callback:
                    progress_callback(arxiv_id, False)
                return False
            finally:
                await asyncio.sleep(self.request_delay)

    def download_source(self, arxiv_id: str) -> Optional[Path]:
        """Download and extract LaTeX source for a single paper.

        Downloads the .tar.gz source archive from ArXiv, extracts it,
        and returns the path to the extracted directory.

        Args:
            arxiv_id: ArXiv paper identifier (e.g., '2301.12345').

        Returns:
            Path to the extracted source directory, or None if download
            failed or source is not available.
        """
        extract_dir = self.sources_dir / arxiv_id.replace("/", "_")

        # Skip if already downloaded
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.debug("Source already exists for %s", arxiv_id)
            return extract_dir

        url = EPRINT_URL.format(arxiv_id=arxiv_id)
        archive_path = self.sources_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"

        try:
            response = self._download_with_retry(url)
            if response is None:
                return None

            # Save the archive
            with open(archive_path, "wb") as f:
                f.write(response.content)

            # Check if it's actually a tar.gz (some papers have single .tex files)
            content_type = response.headers.get("Content-Type", "")
            if "gzip" in content_type or "tar" in content_type:
                return self._extract_archive(archive_path, extract_dir)
            elif "text/plain" in content_type or "tex" in content_type:
                # Single .tex file, not an archive
                extract_dir.mkdir(parents=True, exist_ok=True)
                tex_path = extract_dir / "main.tex"
                tex_path.write_bytes(response.content)
                archive_path.unlink(missing_ok=True)
                return extract_dir
            else:
                # Try to extract anyway — sometimes content-type is unreliable
                return self._extract_archive(archive_path, extract_dir)

        except Exception as e:
            logger.warning("Failed to download source for %s: %s", arxiv_id, e)
            archive_path.unlink(missing_ok=True)
            return None

    def download_pdf(self, arxiv_id: str) -> Optional[Path]:
        """Download PDF for a single paper (fallback when source unavailable).

        Args:
            arxiv_id: ArXiv paper identifier.

        Returns:
            Path to the downloaded PDF, or None if download failed.
        """
        pdf_path = self.pdfs_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        if pdf_path.exists():
            logger.debug("PDF already exists for %s", arxiv_id)
            return pdf_path

        url = PDF_URL.format(arxiv_id=arxiv_id)

        try:
            response = self._download_with_retry(url)
            if response is None:
                return None

            with open(pdf_path, "wb") as f:
                f.write(response.content)

            return pdf_path

        except Exception as e:
            logger.warning("Failed to download PDF for %s: %s", arxiv_id, e)
            return None

    async def batch_download_sources(
        self, arxiv_ids: list[str], progress_callback=None
    ) -> dict[str, Optional[Path]]:
        """Download sources for multiple papers concurrently.

        Uses async HTTP with a semaphore to limit concurrency and
        a delay between requests to respect rate limits.

        Args:
            arxiv_ids: List of ArXiv IDs to download.
            progress_callback: Optional callable(arxiv_id, success) for progress tracking.

        Returns:
            Dictionary mapping arxiv_id to extracted source path (or None on failure).
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "arxiv-section-extractor/0.1 (research project)"},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as session:
            tasks = [
                self._async_download_one(
                    session, semaphore, arxiv_id, progress_callback
                )
                for arxiv_id in arxiv_ids
            ]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for arxiv_id, result in zip(arxiv_ids, completed):
                if isinstance(result, Exception):
                    logger.warning("Batch download failed for %s: %s", arxiv_id, result)
                    results[arxiv_id] = None
                else:
                    results[arxiv_id] = result

        return results

    async def _async_download_one(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        arxiv_id: str,
        progress_callback=None,
    ) -> Optional[Path]:
        """Download a single source file asynchronously.

        Args:
            session: aiohttp session to use.
            semaphore: Concurrency limiter.
            arxiv_id: Paper ID to download.
            progress_callback: Optional progress reporter.

        Returns:
            Path to extracted source, or None.
        """
        async with semaphore:
            extract_dir = self.sources_dir / arxiv_id.replace("/", "_")
            if extract_dir.exists() and any(extract_dir.iterdir()):
                if progress_callback:
                    progress_callback(arxiv_id, True)
                return extract_dir

            url = EPRINT_URL.format(arxiv_id=arxiv_id)
            archive_path = self.sources_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"

            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(
                            "HTTP %d for source %s", response.status, arxiv_id
                        )
                        if progress_callback:
                            progress_callback(arxiv_id, False)
                        return None

                    content = await response.read()
                    with open(archive_path, "wb") as f:
                        f.write(content)

                    result = self._extract_archive(archive_path, extract_dir)
                    if progress_callback:
                        progress_callback(arxiv_id, result is not None)
                    return result

            except Exception as e:
                logger.warning("Async download failed for %s: %s", arxiv_id, e)
                archive_path.unlink(missing_ok=True)
                if progress_callback:
                    progress_callback(arxiv_id, False)
                return None
            finally:
                await asyncio.sleep(self.request_delay)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=30))
    def _download_with_retry(self, url: str) -> Optional[requests.Response]:
        """Download a URL with retry logic.

        Args:
            url: URL to download.

        Returns:
            HTTP response object, or None for 4xx errors.
        """
        time.sleep(self.request_delay)
        response = self._session.get(url, timeout=60)

        if response.status_code == 404:
            logger.debug("404 for %s (no source available)", url)
            return None
        if response.status_code == 503:
            retry_after = int(response.headers.get("Retry-After", 30))
            logger.warning("503 from ArXiv, waiting %ds", retry_after)
            time.sleep(retry_after)
            raise requests.exceptions.ConnectionError("503 retry")

        response.raise_for_status()
        return response

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> Optional[Path]:
        """Extract a .tar.gz archive to a directory.

        Args:
            archive_path: Path to the archive file.
            extract_dir: Directory to extract into.

        Returns:
            Path to the extraction directory, or None if extraction failed.
        """
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                # Security: only extract safe members
                safe_members = [
                    m for m in tar.getmembers()
                    if not m.name.startswith(("/", ".."))
                    and ".." not in m.name
                ]
                tar.extractall(path=extract_dir, members=safe_members)
            archive_path.unlink(missing_ok=True)
            return extract_dir
        except (tarfile.TarError, EOFError) as e:
            logger.warning("Failed to extract %s: %s", archive_path, e)
            archive_path.unlink(missing_ok=True)
            return None
