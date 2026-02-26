# ArXiv Section Extractor

Large-scale extraction of structured sections (abstract, introduction, method, results, conclusion) from ArXiv CS papers (2022–2025) into a Parquet dataset.

## Quickstart

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Test on a Single Paper

```bash
python -m src.cli single 2301.12345
```

This downloads the source, parses sections, and prints the results. Great for verifying the pipeline works.

### 3. Fetch Metadata

**Option A — Kaggle dataset (recommended for speed):**
Download the [ArXiv dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv), then:

```bash
python -m src.cli metadata --source kaggle --kaggle-json arxiv-metadata-oai-snapshot.json
```

**Option B — OAI-PMH (complete but slow):**

```bash
python -m src.cli metadata --source oai --date-start 2022-01-01 --date-end 2025-12-31
```

### 4. Run Full Pipeline

```bash
python -m src.cli run --config configs/default.yaml --kaggle-json arxiv-metadata-oai-snapshot.json
```

The pipeline saves checkpoints every 1000 papers and can be resumed if interrupted.

### 5. Load Results

```python
import pandas as pd
df = pd.read_parquet("output/arxiv_cs_sections.parquet")
print(f"Papers: {len(df)}")
print(f"With all sections: {df['extraction_success'].sum()}")
print(df[['arxiv_id', 'title', 'sections_found']].head())
```

## Configuration

Edit `configs/default.yaml` to adjust:
- Date range and categories
- Download concurrency and rate limits
- Number of parallel workers
- LLM fallback settings (model, budget)
- Output path and compression

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details on the pipeline design, component breakdown, resource estimates, and deployment options.

## Project Structure

```
arxiv_section_extractor/
├── configs/
│   └── default.yaml          # Pipeline configuration
├── docs/
│   └── ARCHITECTURE.md       # Detailed architecture docs
├── src/
│   ├── cli.py                # Command-line interface
│   ├── metadata.py           # ArXiv metadata fetching
│   ├── downloader.py         # Source/PDF downloading
│   ├── latex_parser.py       # LaTeX → sections
│   ├── pdf_extractor.py      # PDF → sections (fallback)
│   ├── section_classifier.py # Section type classification
│   ├── pipeline.py           # Orchestration & checkpointing
│   └── models.py             # Data models
├── tests/
├── pyproject.toml
└── README.md
```
