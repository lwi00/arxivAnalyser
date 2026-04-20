# ArXiv Related-Work Analysis

This repository extracts related-work sections from arXiv computer-science papers
and analyzes whether newer related-work writing is easier for language models to
predict than older writing.

The project has two analysis layers:

1. A broad notebook analysis over the extracted corpus, covering readability,
   lexical diversity, passive voice, citation structure, LLM-favored markers,
   GPT-2 perplexity, and a small Binoculars-style detector run.
2. A stricter curated Qwen pass over a balanced, word-count-stratified subset,
   used as the current clean result for model-based predictability.

The current tracked experiment is in `experiments/qwen_curated/`. Raw/generated
corpora and older intermediate outputs have been moved under `local_artifacts/`,
which is ignored by git.

## Research Question

The working hypothesis is that widespread LLM availability after 2022 has
measurably changed the prose style of related-work sections in computer-science
papers. The project does not attempt to classify individual papers as AI-written.
Instead, it looks for aggregate distribution shifts between an older comparison
group and a 2025 group.

The evidence is intentionally multi-axis:

- surface style: readability, lexical diversity, sentence structure, passive voice
- vocabulary markers: high-confidence LLM-favored phrases from prior literature
- structure: word count, paragraph count, citation counts and citation density
- model predictability: causal-LM perplexity and sentence-level burstiness
- relative model behavior: a Binoculars-style base/instruct comparison

## Current Curated Qwen Result

The main tracked result file is:

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

This result is deliberately conservative: the manifest is balanced by year group
and matched by word-count stratum before scoring.

## Full-Corpus Notebook Findings

The original `analysis.ipynb` ran over the larger extracted corpus before the
cleanup:

- `8,692` related-work text files
- `8,683` papers before filtering
- `8,666` papers after removing sections with fewer than 50 words
- `4,610` older-group papers
- `4,056` 2025 papers

The exported full-corpus metrics were moved to:

```text
local_artifacts/2026-04-20-cleanup/old_outputs/related_works_metrics.csv
```

That file is ignored by git, but it preserves the notebook output locally.

### Style Shift

The notebook found a strong style shift between the two groups. Mean sentence
length is effectively unchanged, but the rest of the linguistic metrics move:

| metric | older mean | 2025 mean | change | effect |
| --- | ---: | ---: | ---: | ---: |
| Mean sentence length | 19.89 | 19.77 | -0.6% | null |
| MATTR lexical diversity | 0.796 | 0.820 | +3.0% | medium |
| Hapax ratio | 0.311 | 0.346 | +11.4% | medium |
| Flesch reading ease | 19.33 | 10.34 | -46.5% | large |
| Gunning Fog | 19.20 | 20.71 | +7.8% | medium-large |
| Passive voice ratio | 0.279 | 0.203 | -27.3% | medium |

The notebook interpretation is that 2025 related-work prose is not simply
"flatter" or shorter. It is denser, harder to read, more lexically varied, and
less passive.

### LLM-Favored Marker Shift

The aggregate high-confidence LLM-marker score rose from `0.048` to `0.276`
markers per 1,000 words, a `+471.6%` increase. The notebook reports a
Mann-Whitney p-value of `2.46e-67`.

Examples of markers that increased sharply in 2025 include:

- `additionally`
- `furthermore`
- `notably`
- `intricate`
- `landscape`
- `underscores`
- `in the realm of`
- `plays a crucial/pivotal role`

The marker analysis is phrase-level evidence; it is not a detector by itself.
Its value is that it points in the same direction as the broader style metrics.

### Structural Stability

The section container changed much less than the prose:

| metric | older mean | 2025 mean | change |
| --- | ---: | ---: | ---: |
| Word count | 623.15 | 588.11 | -5.6% |
| Paragraph count | 9.28 | 8.55 | -7.9% |
| Citation count | 18.85 | 18.69 | -0.9% |
| Citations per sentence | 0.661 | 0.695 | +5.1% |
| Citations per 100 words | 3.403 | 3.599 | +5.8% |

Citation counts are essentially stable, and citation-introduction patterns moved
only modestly. The main shift is in the prose between citations, not in the
basic structure of related-work sections.

### Detector-Style Results

The original notebook merged older detector runs:

| detector | older mean | 2025 mean | change | note |
| --- | ---: | ---: | ---: | --- |
| GPT-2-medium log perplexity | 3.355 | 3.341 | -0.4% | significant but tiny |
| Sentence burstiness std | 0.805 | 0.796 | -1.2% | significant but tiny |
| Binoculars score, 500/500 sample | 0.9701 | 0.9736 | +0.4% | small-to-medium signal |

The notebook conclusion was that single-model perplexity and burstiness are weak
signals at this scale, while the relative base/instruct Binoculars-style score
is more informative. The later curated Qwen pass confirms the single-model
direction on a balanced manifest, but still shows only a small effect size.

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

### 2. Notebook Feature Extraction

`analysis.ipynb` builds paper-level features from the extracted text:

- sentence and word counts
- paragraph counts
- citation counts and citation density
- Flesch reading ease
- Gunning Fog index
- type-token ratio, MATTR, and hapax ratio
- passive voice ratio
- LLM-marker frequencies
- optional merged perplexity, burstiness, and Binoculars metrics

The notebook uses Mann-Whitney U tests, Welch's t-tests, Cohen's d,
rank-biserial correlation, and Benjamini-Hochberg FDR correction across the
master metric table.

### 3. Curated Manifest

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

### 4. Language-Model Scoring

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
