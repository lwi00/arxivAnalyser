"""Standalone compute script for perplexity_analysis.ipynb.

Produces the same perplexity_metrics.csv that perplexity_analysis.ipynb expects,
so the notebook can be re-run later with instant cache hits instead of ~6h of GPU.

Resumable: checkpoints every 500 papers to perplexity_metrics.csv.partial.
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

import nltk
from nltk.tokenize import sent_tokenize

# --- Config (must match perplexity_analysis.ipynb) ---
DATA_2015 = "./data/txt"
DATA_2025 = "./data-2025/txt"
MODEL_NAME = "gpt2-medium"
MAX_LEN = 1024
STRIDE = 512
MIN_SENT_TOKENS = 5
CACHE_PATH = pathlib.Path("perplexity_metrics.csv")
PARTIAL_PATH = pathlib.Path("perplexity_metrics.csv.partial")
CHECKPOINT_EVERY = 500
LOG_EVERY = 50

warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # already-cached punkt is enough


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Corpus loading (same as notebook) ---
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


# --- Scoring helpers (same as notebook) ---
def load_model(device: torch.device):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # fp16 on CUDA gives ~2-3x speedup and halves VRAM; loss stays stable because
    # GPT-2 log-perplexity on English is in the 2-5 range (no underflow).
    # MPS/CPU stays in fp32 — fp16 on MPS has known instability.
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
        log(f"  model dtype: float16 (CUDA)")
    else:
        model = model.to(device)
        log(f"  model dtype: float32 ({device.type})")
    model.eval()
    with torch.no_grad():
        warm = tok("warm up forward pass", return_tensors="pt").to(device)
        model(**warm, labels=warm.input_ids)
    return tok, model


@torch.no_grad()
def doc_log_ppl(text: str, tok, model, device):
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


@torch.no_grad()
def sentence_log_ppls(text: str, tok, model, device):
    out = []
    for s in sent_tokenize(text):
        if not s.strip():
            continue
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(device)
        if ids.size(1) < MIN_SENT_TOKENS:
            continue
        o = model(ids, labels=ids)
        out.append(o.loss.float().item())
    return out


def compute_paper_metrics(text: str, tok, model, device) -> dict:
    log_ppl, n_tok = doc_log_ppl(text, tok, model, device)
    sent_lps = sentence_log_ppls(text, tok, model, device)
    n_s = len(sent_lps)
    if n_s >= 2:
        arr = np.asarray(sent_lps, dtype=np.float64)
        b_std = float(arr.std(ddof=1))
        mean_lp = float(arr.mean())
        b_cv = float(b_std / mean_lp) if mean_lp != 0 else float("nan")
    else:
        b_std = float("nan")
        b_cv = float("nan")
    return {
        "perplexity": float(math.exp(log_ppl)) if np.isfinite(log_ppl) else float("nan"),
        "log_perplexity": log_ppl,
        "burstiness_std": b_std,
        "burstiness_cv": b_cv,
        "n_sentences_scored": n_s,
        "n_tokens_scored": n_tok,
    }


def main() -> int:
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, dtype={"arxiv_id": str, "year_group": str})
        log(f"{CACHE_PATH} already exists with {len(df)} rows — nothing to do.")
        return 0

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log(f"device={device} torch={torch.__version__}")

    log("Building corpus ...")
    df_papers = build_df_papers()
    log(f"df_papers: {len(df_papers)} rows ({df_papers['year_group'].value_counts().to_dict()})")

    log(f"Loading {MODEL_NAME} ...")
    t_load = time.time()
    tok, model = load_model(device)
    log(f"Model loaded in {time.time() - t_load:.1f}s ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Resume from partial if present. dtype={str,str} is critical: CSV roundtrip
    # would otherwise cast numeric-looking arxiv_ids (e.g. "1408.2103") to float64,
    # breaking the merge in analysis.ipynb.
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
        m = compute_paper_metrics(row["text_clean"], tok, model, device)
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
            log(f"  {i + 1}/{len(df_todo)}  rate={rate:.2f} pap/s  elapsed={elapsed / 60:.1f}m  eta={eta_min:.1f}m")
            last_log = now
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(PARTIAL_PATH, index=False)
            log(f"  checkpoint: wrote {len(results)} rows to {PARTIAL_PATH}")

    df_ppl = pd.DataFrame(results)
    df_ppl.to_csv(CACHE_PATH, index=False)
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
    total_min = (time.time() - t0) / 60
    log(f"Done. Wrote {len(df_ppl)} rows to {CACHE_PATH} in {total_min:.1f} min")

    log("Group means:")
    means = df_ppl.groupby("year_group")[["perplexity", "log_perplexity", "burstiness_std", "burstiness_cv"]].mean()
    print(means.round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
