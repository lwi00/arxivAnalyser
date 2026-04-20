"""Standalone compute script for a simplified Binoculars-style two-model score.

Binoculars (Hans et al. 2024, arXiv:2401.12070) detects LLM-generated text by
comparing two LMs from the same family — typically a base model and its
instruction-tuned sibling. Aligned/instruct models are *much* more confident
on LLM-style text (low perplexity), while the base model is less so. Human
text shrinks that gap.

The original paper's score is `log_ppl(M_base) / X-ppl(M_base || M_instruct)`,
where X-ppl is a token-by-token cross-entropy between the two models' logits.
We use a *simplified* version: `log_ppl(M_base) / log_ppl(M_instruct)`.
It needs only two independent forward passes per doc (same as one pass under
each model) and captures the same intuition: text that is much easier for the
instruct model than the base model is plausibly LLM-shaped.

Model pair: Qwen2.5-0.5B + Qwen2.5-0.5B-Instruct.
- Same tokenizer & vocab, so per-token log-ppls are directly comparable.
- 0.5B params each → ~1 GB in fp16 per model, fits MPS / any modern GPU.
- Trained on data through ~2024, so post-LLM contamination exists, but the
  *ratio* is what matters: alignment-induced distribution shift is largely
  orthogonal to whatever pretraining contamination is shared by both models.

Default mode: 500/group stratified subsample (seed=42), matching the
perplexity preview in perplexity_analysis.ipynb cell 9. Set
SUBSAMPLE_PER_GROUP = None below for the full corpus.

Resumable: checkpoints every 500 papers to binoculars_metrics.csv.partial.
"""
from __future__ import annotations

import math
import pathlib
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
DATA_2015 = "./data/txt"
DATA_2025 = "./data-2025/txt"
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 1024
STRIDE = 512
CACHE_PATH = pathlib.Path("binoculars_metrics.csv")
PARTIAL_PATH = pathlib.Path("binoculars_metrics.csv.partial")
CHECKPOINT_EVERY = 500
LOG_EVERY = 50

# Subsample mode: if set, draw N papers per year_group with seed=SEED.
# Set SUBSAMPLE_PER_GROUP = None to score the full corpus.
# Seed and helper are byte-identical to perplexity_analysis.ipynb cell 9 so the
# *same* 1,000 papers used for the perplexity preview are scored here too,
# letting us cross-compare binoculars vs log_ppl row-by-row.
SUBSAMPLE_PER_GROUP: int | None = 500
SEED = 42

warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Corpus loading (identical to run_perplexity.py) ---
CIT_PATTERN = re.compile(r"<cit\.>")
REF_PATTERN = re.compile(r"<ref>")
GRAPHICS_PATTERN = re.compile(r"<graphics>")


def load_corpus(base_path: str, year_group: str) -> list[dict]:
    records = []
    base = pathlib.Path(base_path)
    assert base.exists(), f"Directory not found: {base}"
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        txt_files = list(entry.glob("*.txt"))
        subdirs = [d for d in entry.iterdir() if d.is_dir()]
        if txt_files:
            arxiv_id = entry.name
            for txt_file in sorted(txt_files):
                records.append({
                    "arxiv_id": arxiv_id,
                    "year_group": year_group,
                    "raw_text": txt_file.read_text(encoding="utf-8", errors="replace"),
                    "filename": txt_file.name,
                })
        for sub in subdirs:
            arxiv_id = f"{entry.name}/{sub.name}"
            for txt_file in sorted(sub.glob("*.txt")):
                records.append({
                    "arxiv_id": arxiv_id,
                    "year_group": year_group,
                    "raw_text": txt_file.read_text(encoding="utf-8", errors="replace"),
                    "filename": txt_file.name,
                })
    return records


def build_df_papers() -> pd.DataFrame:
    df_files = pd.DataFrame(load_corpus(DATA_2015, "2015") + load_corpus(DATA_2025, "2025"))
    df_papers = (
        df_files
        .sort_values(["arxiv_id", "filename"])
        .groupby(["arxiv_id", "year_group"], as_index=False)
        .agg(text=("raw_text", "\n\n".join), n_files=("filename", "count"))
    )
    df_papers["text_clean"] = (
        df_papers["text"]
        .str.replace(GRAPHICS_PATTERN, " ", regex=True)
        .str.replace(REF_PATTERN, " ", regex=True)
        .str.replace(CIT_PATTERN, " CITATION ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df_papers["_wc"] = df_papers["text_clean"].str.split().str.len()
    df_papers = df_papers[df_papers["_wc"] >= 50].reset_index(drop=True)
    return df_papers


def stratified_sample(df: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    """Draw n_per_group rows per year_group with a fixed seed.

    Byte-identical to perplexity_analysis.ipynb cell 9 so the SAME 500 papers
    per year_group are picked here and there — guaranteeing apples-to-apples
    comparison between the binoculars score and log_perplexity / burstiness.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for grp, sub in df.groupby("year_group"):
        k = min(n_per_group, len(sub))
        idx = rng.choice(sub.index.values, size=k, replace=False)
        parts.append(sub.loc[idx])
    return pd.concat(parts).sort_index()


# --- Model loading ---
def load_model(name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
        log(f"  {name}: float16 on cuda")
    else:
        model = model.to(device)
        log(f"  {name}: float32 on {device.type}")
    model.eval()
    with torch.no_grad():
        warm = tok("warm up forward pass", return_tensors="pt").to(device)
        model(**warm, labels=warm.input_ids)
    return tok, model


# --- Doc-level sliding-window log-ppl, identical recipe to run_perplexity.py ---
@torch.no_grad()
def doc_log_ppl(text: str, tok, model, device) -> tuple[float, int]:
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    seq_len = ids.size(1)
    if seq_len < 2:
        return float("nan"), 0
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, STRIDE):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        chunk = ids[:, begin:end]
        target = chunk.clone()
        target[:, :-trg_len] = -100
        out = model(chunk, labels=target)
        nlls.append(out.loss.float() * trg_len)
        prev_end = end
        if end == seq_len:
            break
    avg_nll = (torch.stack(nlls).sum() / seq_len).item()
    return avg_nll, seq_len


def compute_paper_metrics(text: str, tok_b, mod_b, tok_i, mod_i, device) -> dict:
    base_lp, n_tok_b = doc_log_ppl(text, tok_b, mod_b, device)
    inst_lp, n_tok_i = doc_log_ppl(text, tok_i, mod_i, device)
    if np.isfinite(base_lp) and np.isfinite(inst_lp) and inst_lp != 0:
        score = base_lp / inst_lp
    else:
        score = float("nan")
    return {
        "base_log_ppl": base_lp,
        "instruct_log_ppl": inst_lp,
        "binoculars_score": score,
        "base_perplexity": float(math.exp(base_lp)) if np.isfinite(base_lp) else float("nan"),
        "instruct_perplexity": float(math.exp(inst_lp)) if np.isfinite(inst_lp) else float("nan"),
        "n_tokens_scored": n_tok_b,
    }


def main() -> int:
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, dtype={"arxiv_id": str, "year_group": str})
        log(f"{CACHE_PATH} already exists with {len(df)} rows — nothing to do.")
        return 0

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    log(f"device={device} torch={torch.__version__}")

    log("Building corpus ...")
    df_papers = build_df_papers()
    log(f"df_papers: {len(df_papers)} rows ({df_papers['year_group'].value_counts().to_dict()})")

    if SUBSAMPLE_PER_GROUP is not None:
        df_papers = stratified_sample(df_papers, SUBSAMPLE_PER_GROUP, SEED).reset_index(drop=True)
        log(f"Subsampled to {SUBSAMPLE_PER_GROUP}/group "
            f"(seed={SEED}): {len(df_papers)} rows "
            f"({df_papers['year_group'].value_counts().to_dict()})")

    log("Loading models ...")
    t_load = time.time()
    tok_b, mod_b = load_model(BASE_MODEL, device)
    tok_i, mod_i = load_model(INSTRUCT_MODEL, device)
    n_params_b = sum(p.numel() for p in mod_b.parameters()) / 1e6
    n_params_i = sum(p.numel() for p in mod_i.parameters()) / 1e6
    log(f"Models loaded in {time.time() - t_load:.1f}s "
        f"(base={n_params_b:.1f}M, instruct={n_params_i:.1f}M)")

    # Tokenizer sanity check — same vocab is required for the score to be apples-to-apples.
    if tok_b.get_vocab() != tok_i.get_vocab():
        log("WARNING: base and instruct tokenizers differ — log-ppls are not directly comparable.")

    # Resume from partial. dtype={str,str} prevents pandas casting "1408.2103"-style
    # arxiv_ids to float64, which would silently break the merge in analysis.ipynb.
    if PARTIAL_PATH.exists():
        df_done = pd.read_csv(PARTIAL_PATH, dtype={"arxiv_id": str, "year_group": str})
        done_keys = set(zip(df_done["arxiv_id"], df_done["year_group"]))
        results = df_done.to_dict("records")
        log(f"Resuming from {PARTIAL_PATH}: {len(df_done)} papers already scored")
    else:
        done_keys = set()
        results = []

    todo_mask = ~df_papers.apply(
        lambda r: (r["arxiv_id"], r["year_group"]) in done_keys, axis=1
    )
    df_todo = df_papers[todo_mask].reset_index(drop=True)
    log(f"To process: {len(df_todo)} papers")

    t0 = time.time()
    last_log = t0
    for i, row in df_todo.iterrows():
        m = compute_paper_metrics(row["text_clean"], tok_b, mod_b, tok_i, mod_i, device)
        results.append({
            "arxiv_id": row["arxiv_id"],
            "year_group": row["year_group"],
            **m,
        })
        if (i + 1) % LOG_EVERY == 0:
            now = time.time()
            rate = LOG_EVERY / (now - last_log) if now > last_log else 0
            elapsed = now - t0
            remaining = len(df_todo) - (i + 1)
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            log(f"  {i + 1}/{len(df_todo)}  rate={rate:.2f} pap/s  "
                f"elapsed={elapsed / 60:.1f}m  eta={eta_min:.1f}m")
            last_log = now
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(PARTIAL_PATH, index=False)
            log(f"  checkpoint: wrote {len(results)} rows to {PARTIAL_PATH}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(CACHE_PATH, index=False)
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
    total_min = (time.time() - t0) / 60
    log(f"Done. Wrote {len(df_out)} rows to {CACHE_PATH} in {total_min:.1f} min")

    log("Group means:")
    means = df_out.groupby("year_group")[
        ["base_log_ppl", "instruct_log_ppl", "binoculars_score"]
    ].mean()
    print(means.round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
