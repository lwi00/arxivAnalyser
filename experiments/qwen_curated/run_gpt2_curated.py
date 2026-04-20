"""Score a curated related-work manifest with a causal LM.

The default model is ``gpt2``. The script is designed for a fresh GPU pod:
it auto-selects CUDA, uses half precision on NVIDIA GPUs, checkpoints partial
results, and can resume from an interrupted run.
"""

from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer


CIT_PATTERN = re.compile(r"<cit\.>")
REF_PATTERN = re.compile(r"<ref>")
GRAPHICS_PATTERN = re.compile(r"<graphics>")
MIN_SENT_TOKENS = 5


warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-2-style perplexity on a manifest.")
    parser.add_argument("--manifest", default="curated_manifest_gpt2.csv")
    parser.add_argument("--output", default="gpt2_curated_metrics.csv")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--data-2015", default="data/txt")
    parser.add_argument("--data-2025", default="data-2025/txt")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16, help="Window/sentence batch size")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--doc-only", action="store_true", help="Skip sentence burstiness")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N manifest rows")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_sentence_tokenizer():
    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        for pkg in ["punkt_tab", "punkt"]:
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

        return sent_tokenize
    except Exception:
        return lambda text: re.split(r"(?<=[.!?])\s+", text)


def clean_text(text: str) -> str:
    return (
        text.replace("\ufeff", " ")
        .replace("\x00", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def normalize_for_model(text: str) -> str:
    text = clean_text(text)
    text = GRAPHICS_PATTERN.sub(" ", text)
    text = REF_PATTERN.sub(" ", text)
    text = CIT_PATTERN.sub(" CITATION ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_text(row: pd.Series, data_2015: pathlib.Path, data_2025: pathlib.Path) -> str:
    file_paths = str(row.get("file_paths", "") or "")
    paths = [pathlib.Path(p) for p in file_paths.split("|") if p]
    existing = [p for p in paths if p.exists()]
    if existing:
        return "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in existing)

    base = data_2015 if row["year_group"] == "2015" else data_2025
    paper_dir = base / str(row["arxiv_id"])
    if not paper_dir.exists() and "/" in str(row["arxiv_id"]):
        paper_dir = base.joinpath(*str(row["arxiv_id"]).split("/"))
    txt_files = sorted(paper_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No text files found for {row['year_group']} {row['arxiv_id']}")
    return "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in txt_files)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else None
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.to(device)
    model.eval()
    return tokenizer, model


def chunk_document(input_ids: torch.Tensor, max_length: int, stride: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    seq_len = int(input_ids.numel())
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        chunk = input_ids[begin:end].clone()
        labels = chunk.clone()
        labels[:-trg_len] = -100
        chunks.append((chunk, labels))
        prev_end = end
        if end == seq_len:
            break
    return chunks


def pad_batch(
    items: list[tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(chunk.numel()) for chunk, _ in items)
    input_batch = torch.full((len(items), max_len), pad_token_id, dtype=torch.long)
    labels_batch = torch.full((len(items), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)

    for i, (chunk, labels) in enumerate(items):
        length = int(chunk.numel())
        input_batch[i, :length] = chunk
        labels_batch[i, :length] = labels
        attention_mask[i, :length] = 1

    return (
        input_batch.to(device),
        labels_batch.to(device),
        attention_mask.to(device),
    )


@torch.inference_mode()
def batched_item_nlls(
    items: list[tuple[torch.Tensor, torch.Tensor]],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> list[tuple[float, int]]:
    """Return per-item negative log-likelihood sums and token counts."""
    scores: list[tuple[float, int]] = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        input_batch, labels_batch, attention_mask = pad_batch(
            batch_items, tokenizer.pad_token_id, device
        )
        logits = model(input_batch, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_batch[..., 1:].contiguous()

        losses = torch.nn.functional.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        valid = shift_labels.ne(-100)
        row_nlls = (losses * valid).sum(dim=1).detach().cpu()
        row_tokens = valid.sum(dim=1).detach().cpu()
        scores.extend(
            (float(nll), int(tokens))
            for nll, tokens in zip(row_nlls.tolist(), row_tokens.tolist())
        )

    return scores


def batched_nll(
    items: list[tuple[torch.Tensor, torch.Tensor]],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> tuple[float, int]:
    scores = batched_item_nlls(items, tokenizer, model, device, batch_size)
    total_nll = sum(nll for nll, _ in scores)
    total_tokens = sum(tokens for _, tokens in scores)

    return total_nll, total_tokens


def doc_log_ppl(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    stride: int,
    batch_size: int,
) -> tuple[float, int]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    if ids.numel() < 2:
        return float("nan"), 0
    chunks = chunk_document(ids, max_length=max_length, stride=stride)
    total_nll, total_tokens = batched_nll(chunks, tokenizer, model, device, batch_size)
    if total_tokens == 0:
        return float("nan"), 0
    return total_nll / total_tokens, total_tokens


def sentence_log_ppls(
    text: str,
    sent_tokenize,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> list[float]:
    items: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        ids = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).input_ids[0]
        if ids.numel() < MIN_SENT_TOKENS:
            continue
        items.append((ids, ids.clone()))

    return [
        nll / total_tokens
        for nll, total_tokens in batched_item_nlls(items, tokenizer, model, device, batch_size)
        if total_tokens
    ]


def compute_metrics(
    text: str,
    sent_tokenize,
    tokenizer,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    log_ppl, n_tokens = doc_log_ppl(
        text,
        tokenizer,
        model,
        device,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    burstiness_std = float("nan")
    burstiness_cv = float("nan")
    n_sentences = 0
    if not args.doc_only:
        sentence_scores = sentence_log_ppls(
            text,
            sent_tokenize,
            tokenizer,
            model,
            device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        n_sentences = len(sentence_scores)
        if n_sentences >= 2:
            arr = np.asarray(sentence_scores, dtype=np.float64)
            burstiness_std = float(arr.std(ddof=1))
            mean_score = float(arr.mean())
            burstiness_cv = float(burstiness_std / mean_score) if mean_score else float("nan")

    return {
        "log_perplexity": log_ppl,
        "perplexity": float(math.exp(log_ppl)) if np.isfinite(log_ppl) else float("nan"),
        "burstiness_std": burstiness_std,
        "burstiness_cv": burstiness_cv,
        "n_sentences_scored": n_sentences,
        "n_tokens_scored": n_tokens,
    }


def main() -> int:
    args = parse_args()
    output_path = pathlib.Path(args.output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    manifest = pd.read_csv(args.manifest, dtype={"arxiv_id": str, "year_group": str})
    if args.limit > 0:
        manifest = manifest.head(args.limit).copy()

    device = choose_device()
    log(f"device={device} torch={torch.__version__} transformers={transformers.__version__}")
    log(f"manifest rows={len(manifest)} groups={manifest['year_group'].value_counts().to_dict()}")

    log(f"Loading model {args.model} ...")
    tokenizer, model = load_model(args.model, device)
    log(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    sent_tokenize = load_sentence_tokenizer()

    if output_path.exists():
        done_df = pd.read_csv(output_path, dtype={"arxiv_id": str, "year_group": str})
        log(f"{output_path} already exists with {len(done_df)} rows; nothing to do.")
        return 0

    if partial_path.exists():
        done_df = pd.read_csv(partial_path, dtype={"arxiv_id": str, "year_group": str})
        done_keys = set(zip(done_df["arxiv_id"], done_df["year_group"]))
        results = done_df.to_dict("records")
        log(f"Resuming from {partial_path}: {len(done_df)} rows")
    else:
        done_keys = set()
        results = []

    todo_mask = ~manifest.apply(
        lambda row: (row["arxiv_id"], row["year_group"]) in done_keys,
        axis=1,
    )
    todo = manifest[todo_mask].reset_index(drop=True)
    log(f"To score: {len(todo)} rows")

    data_2015 = pathlib.Path(args.data_2015)
    data_2025 = pathlib.Path(args.data_2025)

    t0 = time.time()
    last_log = t0
    for i, row in todo.iterrows():
        raw_text = load_text(row, data_2015, data_2025)
        text = normalize_for_model(raw_text)
        metrics = compute_metrics(text, sent_tokenize, tokenizer, model, device, args)

        results.append(
            {
                "arxiv_id": row["arxiv_id"],
                "year_group": row["year_group"],
                "model_name": args.model,
                "n_words": int(row.get("n_words", 0)),
                "n_files": int(row.get("n_files", 0)),
                "n_citations": int(row.get("n_citations", 0)),
                "cit_per_100_words": float(row.get("cit_per_100_words", 0.0)),
                "word_stratum": int(row.get("word_stratum", -1)),
                **metrics,
            }
        )

        completed = i + 1
        if completed % args.log_every == 0:
            now = time.time()
            rate = args.log_every / (now - last_log) if now > last_log else 0.0
            remaining = len(todo) - completed
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            log(
                f"{completed}/{len(todo)} rate={rate:.2f} docs/s "
                f"elapsed={(now - t0) / 60:.1f}m eta={eta_min:.1f}m"
            )
            last_log = now

        if completed % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(partial_path, index=False)
            log(f"checkpoint: wrote {len(results)} rows to {partial_path}")

    out = pd.DataFrame(results)
    out.to_csv(output_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    total_min = (time.time() - t0) / 60
    log(f"Done. Wrote {len(out)} rows to {output_path} in {total_min:.1f} min")
    metric_cols = ["log_perplexity", "perplexity", "burstiness_std", "burstiness_cv"]
    print(out.groupby("year_group")[metric_cols].mean().round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
