# Qwen Curated Related-Work Pass

This directory contains the curated 2015/2025 related-work scoring pass used for
the current analysis.

## Files

- `qwen25_05b_curated_metrics.csv`: final metrics from `Qwen/Qwen2.5-0.5B`.
- `curated_manifest_gpt2.csv`: balanced curated manifest used by both GPT-2 and Qwen runs.
- `run_gpt2_curated.py`: generic causal-LM scoring script; despite the name, it accepts any Hugging Face causal LM via `--model`.
- `build_curated_manifest.py`: manifest builder.
- `run_perplexity.py` and `run_binoculars.py`: earlier standalone scoring scripts kept for reference.
- `configs/2025.yaml`: 2025 extraction configuration used for the generated corpus.

## Re-run

Run from the repository root so default data paths resolve correctly:

```bash
python experiments/qwen_curated/run_gpt2_curated.py \
  --manifest experiments/qwen_curated/curated_manifest_gpt2.csv \
  --output experiments/qwen_curated/qwen25_05b_curated_metrics.csv \
  --model Qwen/Qwen2.5-0.5B \
  --batch-size 8
```

The raw/generated corpora used for the completed run were moved to
`local_artifacts/2026-04-20-cleanup/`, which is intentionally ignored by git.
