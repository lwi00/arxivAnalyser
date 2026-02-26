# Architecture Documentation

## Overview

The ArXiv Section Extractor is a pipeline for processing **~150k-200k CS papers** from ArXiv (2022-2025) and extracting their standard sections (abstract, introduction, method, results, conclusion) into a structured Parquet dataset.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PIPELINE FLOW                                │
│                                                                      │
│  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌────────────┐    │
│  │ Metadata │──▶│  Filter   │──▶│  Download   │──▶│   Parse    │──┐ │
│  │ Fetcher  │   │ (HEAD     │   │  LaTeX Src  │   │ & Classify │  │ │
│  └──────────┘   │  probes)  │   └────────────┘   └────────────┘  │ │
│       │         └───────────┘                          │          │ │
│   OAI-PMH /     Keep only       LaTeX .tar.gz     LaTeX Parser   │ │
│   Kaggle JSON   papers with                       Classifier     │ │
│                 LaTeX source                                     │ │
│                                                     ┌─────────┐ │ │
│                                                     │ Output  │◀┘ │
│                                                     │ Parquet │   │
│                                                     └─────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Metadata Fetcher (`src/metadata.py`)

**Purpose:** Collect ArXiv paper IDs and metadata for all CS papers in the target date range.

**Data Sources:**
- **OAI-PMH API** (`http://export.arxiv.org/oai2`): Official bulk metadata harvesting protocol. Returns XML with ~1000 records per page, paginated via resumption tokens. Rate-limited server-side (~3s between requests). Complete but slow (~24-48 hours for full harvest).
- **Kaggle JSON** (`arxiv-metadata-oai-snapshot.json`): Pre-built JSON-lines dump of all ArXiv metadata. Much faster to load (minutes). May lag a few months behind. Recommended for quick starts.

**Output:** `metadata_cache.jsonl` — one JSON object per paper with ID, title, authors, categories, dates, abstract.

### 2. Downloader (`src/downloader.py`)

**Purpose:** Filter and download LaTeX source files from ArXiv.

**Two-phase approach:**

1. **HEAD Probe** (filtering): Before committing to full downloads, sends lightweight HEAD requests to `https://arxiv.org/e-print/{id}`. ArXiv returns different `Content-Type` headers:
   - `application/x-eprint-tar` or `application/gzip` → LaTeX source available ✓
   - `application/pdf` → PDF-only, no source ✗
   - `404` → paper not found ✗
   
   This filters out ~10-15% of papers that lack LaTeX source, saving significant download time and disk space.

2. **Source Download**: For papers that pass the filter, downloads the full `.tar.gz` archive from the e-print endpoint and extracts it.

**Rate Limiting:** 3-second delay between requests. Async batch probing/downloading with configurable concurrency (default: 5).

**S3 Bulk Access** (recommended for full runs): ArXiv provides a requester-pays S3 bucket (`s3://arxiv/`) with all sources. Requires AWS credentials. Much faster than HTTP. Note: S3 doesn't support HEAD-based content type probing, so filtering would need to happen after download.

### 3. LaTeX Parser (`src/latex_parser.py`)

**Purpose:** Extract sections from LaTeX source files.

**Algorithm:**
1. **Find root .tex file** — heuristic search: look for `\documentclass`, then common names (`main.tex`, `paper.tex`), then largest `.tex` file.
2. **Resolve includes** — recursively replace `\input{file}` and `\include{file}` with file contents.
3. **Strip comments** — remove `% ...` lines.
4. **Extract abstract** — find `\begin{abstract}...\end{abstract}` environment.
5. **Extract sections** — regex match `\section{Title}` commands, capture content between them.
6. **Clean text** — strip LaTeX commands using `pylatexenc`, then regex cleanup for remaining markup.

**Design Decision: Regex vs. Full Parser**
We use regex-based extraction instead of TexSoup/full LaTeX parsing because:
- Real ArXiv papers frequently contain non-standard LaTeX that breaks strict parsers
- Custom macros, wrong nesting, unusual packages cause parse failures
- Regex is more robust for the messy reality of 200k+ diverse papers
- The accuracy tradeoff is acceptable (we only need section boundaries, not full AST)

**Expected Coverage:** ~85-90% of CS papers have LaTeX source available. Of those, the parser successfully extracts sections from ~95%+, giving an overall yield of ~80-85% of all CS papers in the target range.

### 4. PDF Extractor (`src/pdf_extractor.py`) — Optional

**Status:** Retained in the codebase but **not used** in the default pipeline. Available if you later want to add PDF fallback support (`pip install -e ".[pdf]"`).

**Purpose:** Fallback for papers where LaTeX source is unavailable.

**Why we skip it by default:** By probing source availability upfront and filtering to LaTeX-only papers, we get ~85-90% coverage of CS ArXiv papers with much higher extraction quality. The remaining ~10-15% PDF-only papers would require font-heuristic-based section detection, which is significantly less reliable.

### 5. Section Classifier (`src/section_classifier.py`)

**Purpose:** Map extracted sections to standard types (abstract, introduction, method, results, conclusion).

**Three-tier approach:**

| Tier | Method | Coverage | Cost |
|------|--------|----------|------|
| 1 | Direct keyword match on heading | ~70% | Free |
| 2 | Fuzzy keyword match with confidence scoring | ~20% | Free |
| 3 | LLM classification (Claude Sonnet) | ~10% | ~$0.003/paper |

**Tier 1: Direct Match**
Section headings like "Introduction", "Methodology", "Results" are matched directly via the `SectionType.from_heading()` method.

**Tier 2: Fuzzy Match**
Handles variations like "Our Approach" → method, "Empirical Evaluation" → results, "Concluding Remarks" → conclusion. Uses a scored keyword table with confidence values.

**Tier 3: LLM Fallback**
For headings that don't match any heuristic (e.g., paper-specific names like "The BERT-XL Framework"), sends the heading + first 500 chars of content to Claude for classification. Budget-capped at 10k calls by default.

**Merging:** Some papers split logical sections across multiple `\section{}` commands (e.g., "Method" and "Implementation" are both "method"). The classifier merges duplicate types by concatenating content.

### 6. Pipeline Orchestrator (`src/pipeline.py`)

**Purpose:** Tie everything together with checkpointing and parallelism.

**Processing Model:**
- Papers are processed in batches (default: 1000)
- Within each batch: download sequentially (rate-limited), then parse in parallel using `ProcessPoolExecutor`
- Parquet checkpoints are written after each batch
- Progress tracked in `progress.json` for resume support

**Parallelism:**
- Download: async with semaphore (I/O-bound)
- Parsing: multiprocess pool (CPU-bound)
- LLM calls: sequential (API rate-limited, only in main process)

### 7. Output Format

**Parquet schema:**

| Column | Type | Description |
|--------|------|-------------|
| `arxiv_id` | string | Paper identifier |
| `title` | string | Paper title |
| `authors` | string | Semicolon-separated author list |
| `categories` | string | Comma-separated ArXiv categories |
| `primary_category` | string | Primary category |
| `published` | string | Publication date ISO |
| `updated` | string | Last update date ISO |
| `abstract` | string | Abstract text |
| `introduction` | string | Introduction section text |
| `method` | string | Method/approach section text |
| `results` | string | Results/experiments section text |
| `conclusion` | string | Conclusion section text |
| `extraction_method` | string | latex_parsed / pdf_extracted / llm_classified |
| `extraction_success` | bool | Whether ≥3 standard sections were found |
| `sections_found` | string | Comma-separated list of found section types |
| `raw_section_count` | int | Original number of sections |

**Estimated size:** ~5-15 GB for 200k papers (compressed Snappy).

## Resource Estimates

| Resource | Estimate | Notes |
|----------|----------|-------|
| **Disk (sources)** | ~150-350 GB | LaTeX sources for ~170k papers (after filtering) |
| **Disk (output)** | ~4-12 GB | Compressed Parquet |
| **RAM** | 8-16 GB | For parallel parsing + Parquet writing |
| **Time (HEAD probing)** | 6-12 hours | 200k HEAD requests at 3s delay, concurrency=5 |
| **Time (download)** | 1-5 days | Filtered set (~170k), depends on S3 vs HTTP |
| **Time (parsing)** | 10-20 hours | 4 workers, ~0.5s/paper average |
| **LLM cost** | ~$30 | 10k calls × $0.003/call (Sonnet) |
| **Total time** | 2-7 days | Dominated by download step |

## Deployment Options

### Local Machine
- Works for testing and small batches (<10k papers)
- 16GB RAM, 500GB+ disk recommended
- Use `num_workers: 4` (match CPU cores)

### Cloud (Recommended for Full Run)
- **AWS EC2 c5.2xlarge** (8 vCPU, 16 GB RAM): ~$0.34/hr, ~$60 total
- Use S3 bulk access (same region as ArXiv bucket: us-east-1)
- EBS gp3 500GB for sources storage
- Total cloud cost: ~$100-150 including S3 transfer

### HPC/University Cluster
- Use SLURM job arrays for parallel batch processing
- Shared filesystem for source storage
- Split metadata into chunks, run as array jobs

## Known Limitations

1. **Multi-column layouts** — PDF extraction may merge columns incorrectly
2. **Non-English papers** — Section heading heuristics are English-only
3. **Appendices** — Currently classified as OTHER, not extracted separately
4. **Figures/tables** — Text within figures is not extracted
5. **Very short papers** (<2 pages) — May not have standard section structure
6. **Papers with only \paragraph{}** — No \section{} commands to detect
