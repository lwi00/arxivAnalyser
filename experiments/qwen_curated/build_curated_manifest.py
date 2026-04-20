"""Build a reproducible curated manifest for GPT-2 robustness scoring.

The manifest samples balanced 2015/2025 related-work sections from the existing
plain-text extraction outputs. It filters obvious extraction noise and matches
the two year groups by word-count strata so downstream perplexity comparisons
are less likely to be driven by document length.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


CIT_PATTERN = re.compile(r"<cit\.>")
REF_PATTERN = re.compile(r"<ref>")
GRAPHICS_PATTERN = re.compile(r"<graphics>")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


@dataclass(frozen=True)
class CorpusRecord:
    arxiv_id: str
    year_group: str
    raw_text: str
    file_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced curated manifest from extracted related-work text."
    )
    parser.add_argument("--data-2015", default="data/txt", help="2015 text corpus directory")
    parser.add_argument("--data-2025", default="data-2025/txt", help="2025 text corpus directory")
    parser.add_argument("--output", default="curated_manifest_gpt2.csv", help="Output CSV path")
    parser.add_argument("--min-words", type=int, default=150, help="Minimum cleaned word count")
    parser.add_argument(
        "--max-quantile",
        type=float,
        default=0.99,
        help="Per-year upper word-count quantile to keep",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.55,
        help="Minimum alphabetic character ratio among non-space characters",
    )
    parser.add_argument(
        "--strata",
        type=int,
        default=10,
        help="Number of pooled word-count quantile strata",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=0,
        help="Rows per year group. 0 keeps the largest balanced stratified set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    return parser.parse_args()


def iter_corpus(base_path: str, year_group: str) -> list[CorpusRecord]:
    """Load per-paper text records from the extraction output directory."""
    records: list[CorpusRecord] = []
    base = pathlib.Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue

        txt_files = sorted(entry.glob("*.txt"))
        if txt_files:
            records.append(_record_from_files(entry.name, year_group, txt_files))

        for subdir in sorted(d for d in entry.iterdir() if d.is_dir()):
            nested_files = sorted(subdir.glob("*.txt"))
            if nested_files:
                records.append(
                    _record_from_files(f"{entry.name}/{subdir.name}", year_group, nested_files)
                )

    return records


def _record_from_files(
    arxiv_id: str, year_group: str, txt_files: list[pathlib.Path]
) -> CorpusRecord:
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in txt_files]
    return CorpusRecord(
        arxiv_id=arxiv_id,
        year_group=year_group,
        raw_text="\n\n".join(texts),
        file_paths=[str(p) for p in txt_files],
    )


def clean_text(text: str) -> str:
    return (
        text.replace("\ufeff", " ")
        .replace("\x00", " ")
    )


def normalize_for_counts(text: str) -> str:
    return (
        clean_text(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def compute_metrics(record: CorpusRecord) -> dict:
    text = normalize_for_counts(record.raw_text)
    text_for_words = GRAPHICS_PATTERN.sub(" ", text)
    text_for_words = REF_PATTERN.sub(" ", text_for_words)
    text_for_words = CIT_PATTERN.sub(" CITATION ", text_for_words)
    text_for_words = re.sub(r"\s+", " ", text_for_words).strip()

    words = WORD_PATTERN.findall(text_for_words)
    nonspace = re.sub(r"\s+", "", text_for_words)
    alpha_chars = sum(ch.isalpha() for ch in nonspace)
    alpha_ratio = alpha_chars / len(nonspace) if nonspace else 0.0
    citations = len(CIT_PATTERN.findall(text))

    return {
        "arxiv_id": record.arxiv_id,
        "year_group": record.year_group,
        "n_files": len(record.file_paths),
        "n_chars": len(text),
        "n_words": len(words),
        "n_citations": citations,
        "cit_per_100_words": citations / len(words) * 100 if words else 0.0,
        "alpha_ratio": alpha_ratio,
        "file_paths": "|".join(record.file_paths),
    }


def assign_strata(df: pd.DataFrame, n_strata: int) -> pd.DataFrame:
    out = df.copy()
    n_unique = out["n_words"].nunique()
    q = min(max(1, n_strata), n_unique)
    out["word_stratum"] = pd.qcut(
        out["n_words"],
        q=q,
        labels=False,
        duplicates="drop",
    ).astype(int)
    return out


def largest_balanced_sample(
    df: pd.DataFrame,
    samples_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """Sample equal 2015/2025 counts from each word-count stratum."""
    rng = np.random.default_rng(seed)
    strata = sorted(df["word_stratum"].unique())

    capacities: dict[int, int] = {}
    for stratum in strata:
        sub = df[df["word_stratum"] == stratum]
        counts = sub["year_group"].value_counts()
        capacities[stratum] = int(min(counts.get("2015", 0), counts.get("2025", 0)))

    total_capacity = sum(capacities.values())
    if total_capacity == 0:
        raise ValueError("No balanced samples available after filtering.")

    if samples_per_group <= 0 or samples_per_group >= total_capacity:
        targets = capacities
    else:
        raw = {
            stratum: samples_per_group * capacity / total_capacity
            for stratum, capacity in capacities.items()
        }
        targets = {
            stratum: min(capacities[stratum], int(np.floor(value)))
            for stratum, value in raw.items()
        }
        remainder = samples_per_group - sum(targets.values())
        fractional = sorted(
            strata,
            key=lambda s: (raw[s] - np.floor(raw[s]), capacities[s]),
            reverse=True,
        )
        while remainder > 0:
            progressed = False
            for stratum in fractional:
                if targets[stratum] < capacities[stratum]:
                    targets[stratum] += 1
                    remainder -= 1
                    progressed = True
                    if remainder == 0:
                        break
            if not progressed:
                break

    parts = []
    for stratum, target in targets.items():
        if target <= 0:
            continue
        for year_group in ["2015", "2025"]:
            sub = df[(df["word_stratum"] == stratum) & (df["year_group"] == year_group)]
            chosen = rng.choice(sub.index.to_numpy(), size=target, replace=False)
            parts.append(df.loc[chosen])

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sort_values(["word_stratum", "year_group", "arxiv_id"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()

    records = iter_corpus(args.data_2015, "2015") + iter_corpus(args.data_2025, "2025")
    df = pd.DataFrame(compute_metrics(record) for record in records)
    print(f"Loaded {len(df)} papers: {df['year_group'].value_counts().to_dict()}")

    before = len(df)
    df = df[df["n_words"] >= args.min_words].copy()
    df = df[df["alpha_ratio"] >= args.min_alpha_ratio].copy()
    print(f"Basic filters: {before} -> {len(df)} rows")

    limits = df.groupby("year_group")["n_words"].quantile(args.max_quantile).to_dict()
    df = df[df.apply(lambda row: row["n_words"] <= limits[row["year_group"]], axis=1)].copy()
    print(
        "Length filter limits: "
        + ", ".join(f"{year}<={limit:.0f}" for year, limit in sorted(limits.items()))
    )

    df = assign_strata(df, args.strata)
    sampled = largest_balanced_sample(df, args.samples_per_group, args.seed)
    output = pathlib.Path(args.output)
    sampled.to_csv(output, index=False)

    print(f"Wrote {len(sampled)} rows to {output}")
    print("Group counts:", sampled["year_group"].value_counts().to_dict())
    print("Stratum counts:")
    print(sampled.groupby(["word_stratum", "year_group"]).size().unstack(fill_value=0))
    print("Word-count summary:")
    print(sampled.groupby("year_group")["n_words"].describe().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
