# ArXiv Related-Work Analysis

This repository extracts related-work sections from arXiv computer-science papers
and analyzes whether newer related-work writing is easier for language models to
predict than older writing.

The current usable result is the curated Qwen pass in
`experiments/qwen_curated/`. The raw/generated corpora and older intermediate
outputs have been moved under `local_artifacts/`, which is ignored by git.

## Current Result

The main result file is:

```text
experiments/qwen_curated/qwen25_05b_curated_metrics.csv
```

It contains `7,838` balanced related-work samples:

- `3,919` papers from the older corpus
- `3,919` papers from the 2025 corpus
- matched by word-count strata
- scored with `Qwen/Qwen2.5-0.5B`

Summary of the completed Qwen run:

| year group | mean log perplexity | mean perplexity | burstiness std | burstiness cv |
| --- | ---: | ---: | ---: | ---: |
| 2015 | 3.0061 | 21.2919 | 0.8923 | 0.1958 |
| 2025 | 2.9694 | 20.7033 | 0.8930 | 0.1974 |

Interpretation: Qwen finds the 2025 related-work sections slightly more
predictable than the older matched sample. The direction matches the GPT-2
baseline, but the effect is still small.

## Repository Layout

```text
.
├── analysis.ipynb                         # Original exploratory/statistical notebook
├── cli.py                                 # Top-level extractor CLI
├── configs/
│   └── default.yaml                       # Default extraction configuration
├── data/                                  # Tracked tiny sample data only
├── docs/
│   └── ARCHITECTURE.md                    # Extraction architecture notes
├── experiments/
│   └── qwen_curated/
│       ├── README.md                      # Experiment-specific notes
│       ├── build_curated_manifest.py      # Builds the balanced manifest
│       ├── curated_manifest_gpt2.csv      # Manifest used by GPT-2 and Qwen runs
│       ├── qwen25_05b_curated_metrics.csv # Final Qwen metrics
│       ├── run_gpt2_curated.py            # Generic causal-LM scoring script
│       ├── run_binoculars.py              # Earlier Binoculars-style reference script
│       ├── run_perplexity.py              # Earlier GPT-2-medium reference script
│       └── configs/
│           └── 2025.yaml                  # 2025 corpus extraction configuration
├── local_artifacts/                       # Ignored generated data and old outputs
├── src/                                   # Extractor implementation
└── tests/                                 # Unit tests
```

## Methodology

### 1. Related-Work Extraction

The extraction pipeline downloads arXiv sources, parses LaTeX, classifies
sections, and writes related-work text files. The relevant modules are in `src/`:

- `metadata.py`: metadata fetching
- `downloader.py`: source download and retry handling
- `latex_parser.py`: LaTeX parsing
- `section_classifier.py`: section detection/classification
- `tex_to_txt.py`: text export
- `pipeline.py`: orchestration and checkpointing

The 2025 extraction configuration used for the generated corpus is archived at:

```text
experiments/qwen_curated/configs/2025.yaml
```

The generated raw corpora used for the completed run are not tracked. They are
currently stored locally at:

```text
local_artifacts/2026-04-20-cleanup/corpus/data_txt_untracked
local_artifacts/2026-04-20-cleanup/corpus/data-2025/txt
```

### 2. Curated Manifest

The manifest builder creates a balanced comparison set from the two corpora. It:

- reads related-work `.txt` files from both groups
- removes very short or low-text-quality samples
- caps extreme lengths per year group
- bins papers into pooled word-count strata
- samples the same number from each year group in each stratum

The produced manifest is:

```text
experiments/qwen_curated/curated_manifest_gpt2.csv
```

Despite the filename, the manifest is model-independent. It was first used for
GPT-2, then reused for Qwen so the comparison stayed fixed.

### 3. Language-Model Scoring

The scoring script computes:

- document-level log perplexity
- document-level perplexity
- sentence-level burstiness standard deviation
- sentence-level burstiness coefficient of variation
- token and sentence counts used for scoring

The current run used:

```text
Qwen/Qwen2.5-0.5B
```

on a RunPod H100 GPU. The first full pass used a larger batch size and hit CUDA
OOM during sentence-level scoring near the heavy tail; checkpointing preserved
the completed rows, and the run was resumed with a smaller batch size.

## Reproducing The Qwen Pass

Install the analysis dependencies:

```bash
pip install -e ".[analysis]"
```

Run from the repository root. Because the generated corpora are now archived
under `local_artifacts/`, pass their paths explicitly:

```bash
python experiments/qwen_curated/run_gpt2_curated.py \
  --manifest experiments/qwen_curated/curated_manifest_gpt2.csv \
  --output experiments/qwen_curated/qwen25_05b_curated_metrics.csv \
  --model Qwen/Qwen2.5-0.5B \
  --data-2015 local_artifacts/2026-04-20-cleanup/corpus/data_txt_untracked \
  --data-2025 local_artifacts/2026-04-20-cleanup/corpus/data-2025/txt \
  --batch-size 8 \
  --checkpoint-every 100 \
  --log-every 50
```

The script resumes automatically from `*.partial` files and leaves the final CSV
only after all rows are written.

## Using The Existing Results

For most analysis work, load the final Qwen CSV directly:

```python
import pandas as pd

df = pd.read_csv(
    "experiments/qwen_curated/qwen25_05b_curated_metrics.csv",
    dtype={"arxiv_id": str, "year_group": str},
)

summary = df.groupby("year_group")[
    ["log_perplexity", "perplexity", "burstiness_std", "burstiness_cv"]
].mean()
print(summary)
```

The original `analysis.ipynb` still contains older hard-coded paths. If it is
used again without edits, it will not automatically pick up the moved Qwen
result file or the ignored raw corpora.

## Extraction Pipeline Quickstart

Install the package:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run the extractor CLI against the default configuration:

```bash
python cli.py run --config configs/default.yaml
```

See `docs/ARCHITECTURE.md` for pipeline design notes and operational details.

## Notes

- `local_artifacts/` is intentionally ignored. It holds generated corpora,
  previous CSV outputs, run logs, smoke outputs, notebook backups, and local
  Python caches.
- The tracked `data/` directory only contains a tiny sample. It is not the full
  corpus used for the Qwen pass.
- Some existing tracked files may still be modified or deleted in the working
  tree. Those are not part of the cleaned experiment layout.
