# ArXiv Related-Work Analysis

This repository extracts the *related-work* sections of arXiv computer-science
papers and looks for aggregate writing-style shifts between a pre-LLM (2015)
group and a post-LLM (2025) group. It ships an extraction pipeline
(`src/` + `cli.py`), an analysis notebook (`analysis.ipynb`), and a tracked
language-model scoring experiment under `experiments/qwen_curated/`. The aim
is not to classify individual papers as AI-written - it is to measure how the
distribution of academic prose has moved.

## Research question

Did the widespread availability of large language models after 2022 measurably
change the prose style of related-work sections in arXiv CS papers?

The evidence is collected along four axes so that a shift in one can be
cross-checked against the others:

- **Surface style** - readability, lexical diversity, sentence structure,
  passive voice.
- **Vocabulary markers** - phrases that prior literature flagged as
  LLM-favored ("delve into", "intricate", "underscores", ...).
- **Structure** - word count, paragraph count, citation count, citation
  density.
- **Model predictability** - causal-LM perplexity, sentence-level
  burstiness, and a Binoculars-style base/instruct ratio.

## Corpus

Two groups of related-work sections extracted from arXiv CS papers:

| group | papers (post-filter) | source window |
| --- | ---: | --- |
| older / 2015 | 4,610 | pre-LLM comparison |
| 2025 | 4,056 | post-LLM |
| total | 8,666 | 8,692 raw text files, 8,683 papers before the <50-word filter |

The generated corpora and older intermediate outputs live under
`local_artifacts/`, which is gitignored. The tracked `data/` directory only
contains a tiny sample (one paper, `data/txt/1706.03762/`). The curated,
word-count-balanced scoring manifest (3,919 + 3,919 papers) and the final
Qwen metrics CSV are tracked under `experiments/qwen_curated/`.

## Analysis - Part 1: Simple NLP

`analysis.ipynb` builds per-paper features from the extracted related-work
text using NLTK tokenization and POS tagging, then compares the two year
groups with Mann-Whitney U tests, Welch's t-tests, Cohen's d, rank-biserial
correlation, and Benjamini-Hochberg FDR correction.

### 1.1 Linguistic style

Six core metrics capture readability, lexical variety, and voice. Mean
sentence length barely moves; the rest all shift with medium-to-large effect
sizes.

| metric | older mean | 2025 mean | change | effect |
| --- | ---: | ---: | ---: | ---: |
| Mean sentence length | 19.89 | 19.77 | −0.6% | null |
| MATTR lexical diversity | 0.796 | 0.820 | +3.0% | medium |
| Hapax ratio | 0.311 | 0.346 | +11.4% | medium |
| Flesch reading ease | 19.33 | 10.34 | −46.5% | large |
| Gunning Fog | 19.20 | 20.71 | +7.8% | medium-large |
| Passive voice ratio | 0.279 | 0.203 | −27.3% | medium |

The interpretation is that 2025 related-work prose is not "flatter" or
shorter. It is denser, harder to read, more lexically varied, and less
passive - consistent with what LLM-assisted writing tends to produce.

### 1.2 LLM-favored vocabulary markers

The notebook counts occurrences of 24 phrases flagged by prior work as
over-represented in LLM output, split into a high-confidence list (rare
before 2022, e.g. "delve into", "intricate", "underscores", "plays a crucial
role", "in the realm of") and an elevated-frequency list (existed before but
spiked, e.g. "additionally", "furthermore", "notably", "landscape").

The aggregate high-confidence marker score rose from **0.048** to **0.276**
markers per 1,000 words, a **+471.6%** increase with Mann-Whitney
*p* ≈ 2.46 × 10⁻⁶⁷. Individual phrases with the sharpest rises include
"underscores" (~25×), "intricate" (~12×), and "plays a crucial role" (~7×).
"Additionally" went from appearing in 8.4% of papers to 26.4%.

The marker list is phrase-level evidence, not a detector by itself. Its
value is that it points in the same direction as the surface-style metrics.

### 1.3 Structural stability

The container around the prose barely changed.

| metric | older mean | 2025 mean | change |
| --- | ---: | ---: | ---: |
| Word count | 623.15 | 588.11 | −5.6% |
| Paragraph count | 9.28 | 8.55 | −7.9% |
| Citation count | 18.85 | 18.69 | −0.9% |
| Citations per sentence | 0.661 | 0.695 | +5.1% |
| Citations per 100 words | 3.403 | 3.599 | +5.8% |

Citation counts are essentially flat and citation-introduction patterns
(parenthetical, author-named, verb-led, list-style) moved only modestly.
What changed is the prose between citations, not the scaffolding of the
section.

## Analysis - Part 2: Perplexity and burstiness

If the style shifts above are real, a causal language model should find the
2025 sections slightly more predictable than the older ones. The notebook
and the `experiments/qwen_curated/` scripts test that with three
predictability-style metrics.

- **Log perplexity** - how surprised a causal LM is by the document, per
  token. Lower means more predictable.
- **Sentence burstiness** - standard deviation (and coefficient of
  variation) of sentence-level perplexity across the document. Human text
  tends to be burstier than LLM text.
- **Binoculars-style ratio** - the cross-perplexity of an instruction-tuned
  model against its base model. Less sensitive to raw domain perplexity
  than a single-model score.

### 2.1 GPT-2-medium baseline (full corpus)

From the notebook's earlier full-corpus run:

| detector | older mean | 2025 mean | change | note |
| --- | ---: | ---: | ---: | --- |
| GPT-2-medium log perplexity | 3.355 | 3.341 | −0.4% | significant but tiny |
| Sentence burstiness std | 0.805 | 0.796 | −1.2% | significant but tiny |

Direction is correct (2025 marginally more predictable, marginally less
bursty), but effect sizes are negligible at this scale. GPT-2's vocabulary
is also years behind the 2025 corpus, which biases the absolute numbers.

### 2.2 Qwen2.5-0.5B curated pass (tracked result)

The cleanest, reproducible predictability experiment in this repo. A
balanced manifest (`experiments/qwen_curated/curated_manifest_gpt2.csv`)
pools papers into word-count strata and samples **3,919** from each year
group per stratum, for **7,838** total. Scoring was done with
`Qwen/Qwen2.5-0.5B` on an H100; results are tracked at
`experiments/qwen_curated/qwen25_05b_curated_metrics.csv`.

| year group | mean log perplexity | mean perplexity | burstiness std | burstiness cv |
| --- | ---: | ---: | ---: | ---: |
| 2015 | 3.0061 | 21.2919 | 0.8923 | 0.1958 |
| 2025 | 2.9694 | 20.7033 | 0.8930 | 0.1974 |

Qwen finds the 2025 sections slightly more predictable than the
word-count-matched older sample. Direction matches GPT-2; effect remains
small. This is the current headline number because the manifest is balanced
before scoring, removing section-length as a confound.

### 2.3 Binoculars-style base/instruct ratio

A stratified 500/500 sub-sample was scored with Qwen2.5-0.5B base and
Qwen2.5-0.5B-Instruct via `run_binoculars.py`:

| detector | older mean | 2025 mean | change | note |
| --- | ---: | ---: | ---: | --- |
| Binoculars score | 0.9701 | 0.9736 | +0.4% | Cohen's *d* ≈ −0.36, *p* ≈ 5 × 10⁻⁸ |

This is the only perplexity-class metric with a clear-direction signal at
non-trivial effect size: the instruction-tuned model is relatively more
confident on the 2025 prose than the base model is. Full-corpus Binoculars
is listed as future work.

### 2.4 Takeaway

Single-model perplexity and burstiness are weak signals here. The
linguistic-style and LLM-marker shifts (Part 1) are much larger than
anything the causal LMs expose on their own. The Binoculars-style
base/instruct contrast is the only model-based signal that lines up cleanly
with the style evidence.

## Headline findings

| axis | metric | direction | effect |
| --- | --- | --- | --- |
| surface style | Flesch reading ease | 2025 ↓ | large (−46.5%) |
| surface style | Passive voice ratio | 2025 ↓ | medium (−27.3%) |
| surface style | Gunning Fog | 2025 ↑ | medium-large (+7.8%) |
| surface style | MATTR / hapax | 2025 ↑ | medium |
| vocabulary | LLM-marker score per 1k words | 2025 ↑ | very large (+471.6%) |
| structure | word count, paragraphs, citations | flat | null / tiny |
| LM predictability | GPT-2 log-ppl, burstiness | 2025 ↓ | tiny but directional |
| LM predictability | Qwen2.5-0.5B (curated) log-ppl | 2025 ↓ | small |
| LM predictability | Binoculars base/instruct | 2025 ↑ | small-medium |

Prose style and vocabulary are where the signal lives; structure barely
moves; single-model predictability is weakly in the expected direction; the
relative base/instruct signal is the cleanest LM-based evidence.

## Repository layout

```text
.
├── analysis.ipynb                            # Main notebook: Part 1 + Part 2 analysis
├── cli.py                                    # Extraction CLI entry point
├── configs/
│   └── default.yaml                          # Default extraction configuration
├── data/                                     # Tracked tiny sample only (1706.03762)
├── docs/
│   └── ARCHITECTURE.md                       # Extraction pipeline design notes
├── experiments/
│   └── qwen_curated/
│       ├── README.md                         # Experiment-specific notes
│       ├── build_curated_manifest.py         # Balanced word-count-stratified manifest
│       ├── curated_manifest_gpt2.csv         # Manifest used by GPT-2 and Qwen runs
│       ├── qwen25_05b_curated_metrics.csv    # Tracked Qwen2.5-0.5B metrics (7,838 rows)
│       ├── run_gpt2_curated.py               # Generic causal-LM scoring script
│       ├── run_perplexity.py                 # Earlier GPT-2-medium reference script
│       ├── run_binoculars.py                 # Earlier base/instruct reference script
│       └── configs/2025.yaml                 # 2025 corpus extraction configuration
├── local_artifacts/                          # Gitignored: generated corpora, old outputs
├── src/
│   ├── metadata.py                           # OAI-PMH and Kaggle JSON metadata fetch
│   ├── downloader.py                         # ArXiv source download + HEAD probing
│   ├── latex_parser.py                       # LaTeX section extraction
│   ├── section_classifier.py                 # 3-tier classifier (keyword / fuzzy / LLM)
│   ├── tex_to_txt.py                         # Per-section plain-text export
│   ├── pipeline.py                           # Orchestration, batching, checkpointing
│   └── models.py                             # Data structures
└── tests/                                    # Unit tests for classifier and tex→txt
```

The extractor uses a three-tier section classifier: direct keyword match
(~70% of sections), scored fuzzy match (~20%), Claude Sonnet fallback on the
ambiguous remainder (~10%, budget-capped). The LaTeX parser is intentionally
regex-based so real-world non-standard LaTeX does not break it.

## How to reproduce

### Install

```bash
# Extractor + dev tools
pip install -e ".[dev]"

# Add analysis dependencies (pandas, matplotlib, seaborn, scipy, nltk,
# torch, transformers) for the notebook and LM scoring scripts
pip install -e ".[analysis]"
```

Python 3.10+ is required. The first notebook run will download NLTK data
(handled in the setup cell).

### Run the extractor

Full pipeline (download → LaTeX parse → classify → write Parquet and/or
per-section `.txt`):

```bash
python cli.py run --config configs/default.yaml
```

Fast path for related-work-only text export (no Parquet, no LLM fallback):

```bash
python cli.py related-works --config configs/default.yaml
```

Debug a single paper end-to-end:

```bash
python cli.py single 1706.03762 --use-llm
```

Other commands: `python cli.py convert ...` (batch LaTeX→txt),
`python cli.py metadata --source oai|kaggle ...` (metadata-only fetch).
See `docs/ARCHITECTURE.md` for pipeline design and operational detail.

### Run the notebook

`analysis.ipynb` expects two populated text corpora:

```text
data/txt/           # older / 2015 group
data-2025/txt/      # 2025 group
```

This repo does **not** ship those corpora. You have two options:

1. **Restore** them from `local_artifacts/` if you have the cleanup snapshot
   (paths used by the Qwen run:
   `local_artifacts/2026-04-20-cleanup/corpus/data_txt_untracked` and
   `local_artifacts/2026-04-20-cleanup/corpus/data-2025/txt`).
2. **Regenerate** them by running the extractor with appropriate date
   windows in `configs/default.yaml` (or the archived
   `experiments/qwen_curated/configs/2025.yaml` for the 2025 config).

The notebook also optionally merges `perplexity_metrics.csv` and
`binoculars_metrics.csv` if present; if missing, those cells skip.

### Re-score with Qwen (or any causal LM)

`experiments/qwen_curated/run_gpt2_curated.py` is generic despite its name
and accepts any HuggingFace causal-LM via `--model`. It auto-resumes from
`*.partial` checkpoints.

```bash
python experiments/qwen_curated/run_gpt2_curated.py \
  --manifest experiments/qwen_curated/curated_manifest_gpt2.csv \
  --output   experiments/qwen_curated/qwen25_05b_curated_metrics.csv \
  --model    Qwen/Qwen2.5-0.5B \
  --data-2015 local_artifacts/2026-04-20-cleanup/corpus/data_txt_untracked \
  --data-2025 local_artifacts/2026-04-20-cleanup/corpus/data-2025/txt \
  --batch-size 8 \
  --checkpoint-every 100 \
  --log-every 50
```

To use the tracked result directly without re-scoring:

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

### Tests

```bash
pytest
```

Current coverage: section-classifier heuristics (including regressions on
headings starting with `i`/`v`/`x` so numbered-prefix stripping does not eat
real words) and the filename-sanitizer used by the tex-to-text exporter.

## Limitations

- **No per-category breakdown.** All CS subcategories are pooled; a shift
  could be driven by subfield growth (e.g. more ML papers in 2025) rather
  than writing style.
- **Field growth confound.** The 2025 group is not demographically matched
  to 2015 beyond word-count stratification in the curated pass.
- **GPT-2 vocabulary drift.** Absolute GPT-2 perplexity values are biased
  by the model's age; only Qwen2.5-0.5B is contemporary.
- **Passive-voice heuristic.** POS-based, imperfect - picks up some false
  positives on scientific prose.
- **No causal claim.** This is a distribution-shift study. It does not
  attempt to label individual papers as AI-written or -assisted.
- **Citation handling.** `<cit.>` is normalized to a CITATION token before
  tokenization; citation-introduction pattern classification is regex-based.

## References

- `docs/ARCHITECTURE.md` - extractor pipeline design (metadata fetching,
  LaTeX parsing, 3-tier classifier, checkpointing).
- `experiments/qwen_curated/README.md` - experiment-specific notes and
  manifest construction.
- `TODO.md` - original notebook specification, used as the build target for
  `analysis.ipynb`.
