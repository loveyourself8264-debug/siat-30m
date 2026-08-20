"""Generic Fast Corpus Audit for Siat pretraining sources.

By default streams the full corpus through clean → filter → exact dedup
and tokenizes only a deterministic sample to estimate total Siat tokens.
Full tokenization requires ``--full-token-audit``.

Example::

    python -m data.audit_corpus \\
        --input data/raw/finewiki_ko \\
        --format parquet \\
        --source finewiki_ko \\
        --language ko \\
        --fast \\
        --tokenizer tokenizer/siat-tokenizer.json \\
        --output data/audits/finewiki_ko_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.build_pretraining_data import PIPELINE_VERSION
from data.dedup import ExactDeduper
from data.document import (
    iter_source_documents,
    list_input_files,
    make_document_id,
)
from data.filters import FilterConfig, FilterStats, filter_document
from tokenizer import load_tokenizer

EOS_TOKEN = "<|eos|>"
DEFAULT_SAMPLE_DOCUMENTS = 10_000
DEFAULT_SEED = 42

_MATH_RE = re.compile(
    r"[=^_$∑√∫±≤≥∞≈≠∂∇πθαβγΔΩ]|\\frac|\\sum|\\int|\\sqrt"
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")


@dataclass
class LengthStats:
    """Online length tracking with reservoir sample for percentiles."""

    count: int = 0
    total: int = 0
    min_v: int | None = None
    max_v: int | None = None
    reservoir: list[int] = field(default_factory=list)
    reservoir_max: int = 50_000
    _rng: random.Random = field(default_factory=lambda: random.Random(42))

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.min_v = value if self.min_v is None else min(self.min_v, value)
        self.max_v = value if self.max_v is None else max(self.max_v, value)
        if len(self.reservoir) < self.reservoir_max:
            self.reservoir.append(value)
        else:
            j = self._rng.randint(0, self.count - 1)
            if j < self.reservoir_max:
                self.reservoir[j] = value

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self.reservoir:
            return float("nan")
        xs = sorted(self.reservoir)
        if len(xs) == 1:
            return float(xs[0])
        k = (len(xs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(xs[int(k)])
        return float(xs[f] * (c - k) + xs[c] * (k - f))


def _has_hangul(text: str) -> bool:
    for ch in text:
        if "\uac00" <= ch <= "\ud7a3":
            return True
    return False


def _has_latin(text: str) -> bool:
    return _LATIN_RE.search(text) is not None


def _has_digits(text: str) -> bool:
    return _DIGIT_RE.search(text) is not None


def _has_math_like(text: str) -> bool:
    return _MATH_RE.search(text) is not None


@dataclass
class FloatStats:
    """Online float tracking with reservoir sample for percentiles."""

    count: int = 0
    total: float = 0.0
    min_v: float | None = None
    max_v: float | None = None
    reservoir: list[float] = field(default_factory=list)
    reservoir_max: int = 50_000
    _rng: random.Random = field(default_factory=lambda: random.Random(42))

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_v = value if self.min_v is None else min(self.min_v, value)
        self.max_v = value if self.max_v is None else max(self.max_v, value)
        if len(self.reservoir) < self.reservoir_max:
            self.reservoir.append(value)
        else:
            j = self._rng.randint(0, self.count - 1)
            if j < self.reservoir_max:
                self.reservoir[j] = value

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def percentile(self, p: float) -> float:
        if not self.reservoir:
            return float("nan")
        xs = sorted(self.reservoir)
        if len(xs) == 1:
            return float(xs[0])
        k = (len(xs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(xs[int(k)])
        return float(xs[f] * (c - k) + xs[c] * (k - f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_schema(
    path: Path, fmt: str, text_field: str
) -> list[str] | None:
    """Return parquet column names, or None for non-parquet formats."""
    if fmt.lower() != "parquet":
        return None
    import pyarrow.parquet as pq

    files = list_input_files(path, ".parquet")
    names = list(pq.ParquetFile(files[0]).schema_arrow.names)
    if text_field not in names:
        raise ValueError(
            f"Expected text column {text_field!r} missing. "
            f"Schema columns: {names}"
        )
    return names


def estimate_tokens_from_chars(
    *,
    kept_characters: int,
    kept_documents: int,
    sample_characters: int,
    sample_tokens: int,
    sample_documents: int,
    per_doc_chars_per_token: list[float] | None = None,
) -> dict[str, Any]:
    """Character-based token estimate with a conservative range.

    Content chars/token excludes EOS. Total estimate adds one EOS per kept doc.
    """
    content_tokens = max(0, sample_tokens - sample_documents)
    if sample_characters <= 0 or content_tokens <= 0 or kept_documents <= 0:
        return {
            "chars_per_token": float("nan"),
            "estimated_total_tokens": 0,
            "estimated_token_range": [0, 0],
            "method": "character_based_plus_eos",
            "is_estimate": True,
        }

    chars_per_token = sample_characters / content_tokens
    estimated = int(round(kept_characters / chars_per_token + kept_documents))

    ratios = [
        r for r in (per_doc_chars_per_token or []) if r > 0 and math.isfinite(r)
    ]
    if len(ratios) >= 2:
        xs = sorted(ratios)
        p10 = xs[int((len(xs) - 1) * 0.10)]
        p90 = xs[int((len(xs) - 1) * 0.90)]
        # Higher chars/token → fewer tokens; invert for bounds
        low = int(round(kept_characters / p90 + kept_documents))
        high = int(round(kept_characters / p10 + kept_documents))
        if low > high:
            low, high = high, low
        # Clamp relative width to roughly ±15% around point estimate
        floor = int(round(estimated * 0.85))
        ceil = int(round(estimated * 1.15))
        low = max(low, floor)
        high = min(high, ceil)
        if low > high:
            low, high = floor, ceil
    else:
        low = int(round(estimated * 0.90))
        high = int(round(estimated * 1.10))

    return {
        "chars_per_token": chars_per_token,
        "estimated_total_tokens": estimated,
        "estimated_token_range": [low, high],
        "method": "character_based_plus_eos",
        "is_estimate": True,
    }


def _encode_texts(
    tokenizer: Any,
    texts: list[str],
    *,
    vocab_size: int,
    encode_batch_size: int = 256,
) -> tuple[list[int], int]:
    """Return per-doc token counts (including EOS) and invalid-id count."""
    counts: list[int] = []
    invalid = 0
    for start in range(0, len(texts), encode_batch_size):
        chunk = texts[start : start + encode_batch_size]
        if hasattr(tokenizer, "encode_batch"):
            encodings = tokenizer.encode_batch(chunk)
        else:
            encodings = [tokenizer.encode(t) for t in chunk]
        for enc in encodings:
            ids = enc.ids
            if ids and (min(ids) < 0 or max(ids) >= vocab_size):
                invalid += 1
                bad = next(tid for tid in ids if tid < 0 or tid >= vocab_size)
                raise RuntimeError(
                    f"Invalid token id {bad} (vocab_size={vocab_size})"
                )
            counts.append(len(ids) + 1)  # EOS
    return counts, invalid


def run_corpus_audit(
    *,
    input_path: str | Path = "data/raw/finewiki_ko",
    fmt: str = "parquet",
    source: str = "finewiki_ko",
    language: str = "ko",
    tokenizer_path: str | Path = "tokenizer/siat-tokenizer.json",
    output_path: str | Path = "data/audits/finewiki_ko_audit.json",
    text_field: str = "text",
    batch_size: int = 1024,
    min_chars: int = 32,
    sample_documents: int = DEFAULT_SAMPLE_DOCUMENTS,
    seed: int = DEFAULT_SEED,
    fast: bool = True,
    full_token_audit: bool = False,
    progress_every: int = 50_000,
) -> dict[str, Any]:
    """Fast (default) or full corpus audit through the Siat data pipeline."""
    input_path = Path(input_path)
    tokenizer_path = Path(tokenizer_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    do_full = bool(full_token_audit)
    # --fast is the default path; full_token_audit overrides
    if do_full:
        fast = False

    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Siat tokenizer not found: {tokenizer_path.resolve()}. "
            "Train or provide tokenizer/siat-tokenizer.json before audit."
        )

    suffix = {
        "parquet": ".parquet",
        "jsonl": ".jsonl",
        "txt": ".txt",
    }.get(fmt.lower())
    if suffix is None:
        raise ValueError(f"Unsupported format: {fmt!r}")

    files = list_input_files(input_path, suffix)
    disk_bytes = sum(f.stat().st_size for f in files)
    schema_cols = inspect_schema(input_path, fmt, text_field)

    print(f"Files: {len(files)}", flush=True)
    if schema_cols is not None:
        print(f"Schema columns: {schema_cols}", flush=True)
    print(f"Using text field: {text_field!r}", flush=True)
    print(f"Disk size: {disk_bytes / 1e6:.2f} MB", flush=True)
    print(
        f"Mode: {'FULL token audit' if do_full else 'FAST (sample tokenize)'}",
        flush=True,
    )

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer missing {EOS_TOKEN!r}")
    if eos_id < 0 or eos_id >= vocab_size:
        raise RuntimeError(f"Invalid EOS id {eos_id}")
    tok_fp = sha256_file(tokenizer_path)

    filter_cfg = FilterConfig(min_chars=min_chars)
    filter_stats = FilterStats()
    deduper = ExactDeduper()

    raw_docs = 0
    raw_chars = 0
    parse_errors = 0
    kept_docs = 0
    kept_chars = 0
    hangul_docs = 0
    latin_docs = 0
    digit_docs = 0
    math_docs = 0
    long_char_stats = LengthStats(_rng=random.Random(seed))

    # Language / education metadata (from parquet when present)
    lang_counts: dict[str, int] = {}
    docs_with_language = 0
    lang_score_stats = FloatStats(_rng=random.Random(seed + 3))
    int_score_counts: dict[str, int] = {}
    docs_with_int_score = 0
    score_ge4 = 0
    int_score_5 = 0
    has_language_meta = False
    has_score_meta = False
    has_token_count_meta = False

    # Reservoir of kept items for fast sample tokenize
    sample_n = max(0, sample_documents)
    sample_rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    # Full-mode token accumulation
    full_token_total = 0
    full_tok_stats = LengthStats(_rng=random.Random(seed + 1))
    encode_calls = 0  # texts passed to encode (for tests / reporting)

    t0 = time.perf_counter()
    print("Scanning corpus...", flush=True)

    pending_full: list[str] = []

    def flush_full() -> None:
        nonlocal full_token_total, encode_calls
        if not pending_full:
            return
        counts, _ = _encode_texts(tokenizer, pending_full, vocab_size=vocab_size)
        encode_calls += len(pending_full)
        for n_tok in counts:
            full_token_total += n_tok
            full_tok_stats.add(n_tok)
        pending_full.clear()

    def _progress() -> None:
        if progress_every > 0 and raw_docs % progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  raw={raw_docs:,} kept={kept_docs:,} "
                f"dup={deduper.stats.duplicates:,} ({elapsed:.1f}s)",
                flush=True,
            )

    for doc, err in iter_source_documents(
        input_path,
        source=source,
        fmt=fmt,
        text_field=text_field,
        language=language,
    ):
        if err is not None:
            parse_errors += 1
            continue
        assert doc is not None
        raw_docs += 1
        raw_chars += len(doc.text)

        meta = doc.metadata or {}
        if "language" in meta and meta["language"] is not None:
            has_language_meta = True
            docs_with_language += 1
            lang_key = str(meta["language"])
            lang_counts[lang_key] = lang_counts.get(lang_key, 0) + 1
        if "language_score" in meta and meta["language_score"] is not None:
            try:
                lang_score_stats.add(float(meta["language_score"]))
            except (TypeError, ValueError):
                pass
        int_score_val: int | None = None
        if "int_score" in meta and meta["int_score"] is not None:
            has_score_meta = True
            try:
                int_score_val = int(meta["int_score"])
                docs_with_int_score += 1
                ik = str(int_score_val)
                int_score_counts[ik] = int_score_counts.get(ik, 0) + 1
                if int_score_val >= 4:
                    score_ge4 += 1
                if int_score_val == 5:
                    int_score_5 += 1
            except (TypeError, ValueError):
                int_score_val = None
        ds_token_count: int | None = None
        if "token_count" in meta and meta["token_count"] is not None:
            has_token_count_meta = True
            try:
                ds_token_count = int(meta["token_count"])
            except (TypeError, ValueError):
                ds_token_count = None
        score_val = meta.get("score")

        kept, _reason = filter_document(doc, filter_cfg, filter_stats)
        if kept is None:
            _progress()
            continue

        kept.document_id = make_document_id(kept.source, kept.text)
        unique = deduper.consider(kept)
        if unique is None:
            _progress()
            continue

        text = unique.text
        n_chars = len(text)
        kept_docs += 1
        kept_chars += n_chars
        long_char_stats.add(n_chars)

        if _has_hangul(text):
            hangul_docs += 1
        if _has_latin(text):
            latin_docs += 1
        if _has_digits(text):
            digit_docs += 1
        if _has_math_like(text):
            math_docs += 1

        item = {
            "text": text,
            "int_score": int_score_val,
            "token_count": ds_token_count,
            "score": score_val,
            "language": meta.get("language"),
        }

        if do_full:
            pending_full.append(text)
            if len(pending_full) >= 256:
                flush_full()
        elif sample_n > 0:
            if len(reservoir) < sample_n:
                reservoir.append(item)
            else:
                j = sample_rng.randint(0, kept_docs - 1)
                if j < sample_n:
                    reservoir[j] = item

        _progress()

    if do_full:
        flush_full()

    # Sample tokenize (fast mode)
    sample_items: list[dict[str, Any]] = list(reservoir) if not do_full else []
    if do_full:
        sample_token_counts: list[int] = []
        sample_chars = 0
        sample_tokens = 0
        sample_texts: list[str] = []
    else:
        n_sample = min(sample_n, kept_docs, len(reservoir))
        sample_items = reservoir[:n_sample]
        sample_texts = [it["text"] for it in sample_items]
        sample_chars = sum(len(t) for t in sample_texts)
        sample_token_counts, _ = _encode_texts(
            tokenizer, sample_texts, vocab_size=vocab_size
        )
        encode_calls += len(sample_texts)
        sample_tokens = sum(sample_token_counts)

    sample_tok_stats = LengthStats(_rng=random.Random(seed + 2))
    per_doc_cpt: list[float] = []
    for text, n_tok in zip(sample_texts, sample_token_counts):
        sample_tok_stats.add(n_tok)
        content = n_tok - 1
        if content > 0:
            per_doc_cpt.append(len(text) / content)

    # Per-int_score sample breakdown + dataset token_count comparison
    by_int: dict[str, dict[str, float]] = {}
    sample_dataset_tokens = 0
    sample_dataset_token_docs = 0
    for it, n_tok in zip(sample_items, sample_token_counts):
        key = (
            str(it["int_score"])
            if it.get("int_score") is not None
            else "unknown"
        )
        bucket = by_int.setdefault(
            key,
            {"documents": 0, "characters": 0, "siat_tokens": 0},
        )
        bucket["documents"] += 1
        bucket["characters"] += len(it["text"])
        bucket["siat_tokens"] += n_tok
        if it.get("token_count") is not None:
            sample_dataset_tokens += int(it["token_count"])
            sample_dataset_token_docs += 1

    sample_by_int_score: dict[str, Any] = {}
    for key, b in sorted(by_int.items(), key=lambda x: x[0]):
        docs_b = int(b["documents"])
        chars_b = b["characters"]
        toks_b = b["siat_tokens"]
        content_b = max(0.0, toks_b - docs_b)
        sample_by_int_score[key] = {
            "documents": docs_b,
            "average_chars": chars_b / docs_b if docs_b else 0.0,
            "average_siat_tokens": toks_b / docs_b if docs_b else 0.0,
            "chars_per_token": (
                chars_b / content_b if content_b > 0 else float("nan")
            ),
        }

    dataset_token_comparison: dict[str, Any] | None = None

    if do_full:
        available_tokens = full_token_total
        estimate_block = {
            "chars_per_token": (
                kept_chars / max(1, available_tokens - kept_docs)
                if kept_docs
                else float("nan")
            ),
            "estimated_total_tokens": available_tokens,
            "estimated_token_range": [available_tokens, available_tokens],
            "method": "full_tokenization",
            "is_estimate": False,
        }
        sample_documents_out = 0
        sample_chars_out = 0
        sample_tokens_out = 0
        avg_tok = full_tok_stats.mean()
        med_tok = full_tok_stats.percentile(50)
        p90_tok = full_tok_stats.percentile(90)
        p95_tok = full_tok_stats.percentile(95)
        p99_tok = full_tok_stats.percentile(99)
        min_tok = full_tok_stats.min_v
        max_tok = full_tok_stats.max_v
        chars_per_token_out = estimate_block["chars_per_token"]
    else:
        estimate_block = estimate_tokens_from_chars(
            kept_characters=kept_chars,
            kept_documents=kept_docs,
            sample_characters=sample_chars,
            sample_tokens=sample_tokens,
            sample_documents=len(sample_texts),
            per_doc_chars_per_token=per_doc_cpt,
        )
        available_tokens = estimate_block["estimated_total_tokens"]
        sample_documents_out = len(sample_texts)
        sample_chars_out = sample_chars
        sample_tokens_out = sample_tokens
        avg_tok = sample_tok_stats.mean()
        med_tok = sample_tok_stats.percentile(50)
        p90_tok = sample_tok_stats.percentile(90)
        p95_tok = sample_tok_stats.percentile(95)
        p99_tok = sample_tok_stats.percentile(99)
        min_tok = sample_tok_stats.min_v
        max_tok = sample_tok_stats.max_v
        chars_per_token_out = estimate_block["chars_per_token"]

    if sample_dataset_token_docs > 0 and sample_documents_out > 0:
        avg_ds = sample_dataset_tokens / sample_dataset_token_docs
        avg_siat = sample_tokens_out / sample_documents_out
        dataset_token_comparison = {
            "sample_docs_with_token_count": sample_dataset_token_docs,
            "dataset_token_count_total": sample_dataset_tokens,
            "dataset_tokens_per_document": avg_ds,
            "siat_tokens_per_document": avg_siat,
            "approx_ratio_siat_over_dataset": (
                avg_siat / avg_ds if avg_ds > 0 else float("nan")
            ),
            "note": (
                "Dataset token_count is not Siat tokens; comparison is audit-only."
            ),
        }

    elapsed = time.perf_counter() - t0
    filtered_total = sum(filter_stats.reasons.values())
    keep_rate = kept_docs / raw_docs if raw_docs else 0.0

    long_10k = sum(1 for v in long_char_stats.reservoir if v >= 10_000)
    long_50k = sum(1 for v in long_char_stats.reservoir if v >= 50_000)
    if long_char_stats.count > long_char_stats.reservoir_max:
        scale = long_char_stats.count / len(long_char_stats.reservoir)
        long_10k_est = int(round(long_10k * scale))
        long_50k_est = int(round(long_50k * scale))
    else:
        long_10k_est = long_10k
        long_50k_est = long_50k

    en_docs = lang_counts.get("en", 0)
    language_statistics: dict[str, Any] | None = None
    if has_language_meta or lang_score_stats.count > 0:
        language_statistics = {
            "documents_with_language_field": docs_with_language,
            "language_counts": dict(sorted(lang_counts.items())),
            "english_ratio": (
                en_docs / docs_with_language if docs_with_language else 0.0
            ),
            "language_score": {
                "count": lang_score_stats.count,
                "mean": lang_score_stats.mean(),
                "median": lang_score_stats.percentile(50),
                "p05": lang_score_stats.percentile(5),
                "p10": lang_score_stats.percentile(10),
                "minimum": lang_score_stats.min_v,
                "maximum": lang_score_stats.max_v,
            },
            "note": "Sanity only; not used for filtering or deletion.",
        }

    education_score_statistics: dict[str, Any] | None = None
    if has_score_meta and docs_with_int_score > 0:
        by_score = {
            k: {
                "documents": v,
                "ratio": v / docs_with_int_score,
            }
            for k, v in sorted(
                int_score_counts.items(),
                key=lambda x: int(x[0]) if x[0].lstrip("-").isdigit() else x[0],
            )
        }
        education_score_statistics = {
            "documents_with_int_score": docs_with_int_score,
            "by_int_score": by_score,
            "score_ge_4_documents": score_ge4,
            "score_ge_4_ratio": score_ge4 / docs_with_int_score,
            "int_score_5_documents": int_score_5,
            "int_score_5_ratio": int_score_5 / docs_with_int_score,
            "sample_by_int_score": sample_by_int_score,
            "note": "No score threshold applied in this audit.",
        }

    report: dict[str, Any] = {
        "source": source,
        "language": language,
        "pipeline_version": PIPELINE_VERSION,
        "unicode_normalization": "NFC",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if do_full else "fast",
        "files": [f.name for f in files],
        "disk_bytes": disk_bytes,
        "raw_documents": raw_docs,
        "kept_documents": kept_docs,
        "filtered_documents": filtered_total,
        "duplicates_removed": deduper.stats.duplicates,
        "raw_characters": raw_chars,
        "kept_characters": kept_chars,
        "sample_documents": sample_documents_out,
        "sample_characters": sample_chars_out,
        "sample_tokens": sample_tokens_out,
        "chars_per_token": chars_per_token_out,
        "estimated_total_tokens": estimate_block["estimated_total_tokens"],
        "estimated_token_range": estimate_block["estimated_token_range"],
        "tokenizer_fingerprint": tok_fp,
        "language_statistics": language_statistics,
        "education_score_statistics": education_score_statistics,
        "dataset_token_comparison": dataset_token_comparison,
        "input": {
            "path": str(input_path.as_posix()),
            "format": fmt,
            "files": [f.name for f in files],
            "file_count": len(files),
            "disk_bytes": disk_bytes,
            "disk_mb": disk_bytes / 1e6,
            "schema_columns": schema_cols,
            "text_field": text_field,
            "raw_documents": raw_docs,
            "raw_characters": raw_chars,
            "average_chars_per_document": (
                raw_chars / raw_docs if raw_docs else 0.0
            ),
            "parse_errors": parse_errors,
            "has_token_count_field": has_token_count_meta,
        },
        "cleaning": {
            "filter_config": {
                "min_chars": filter_cfg.min_chars,
                "max_repeat_ratio": filter_cfg.max_repeat_ratio,
                "min_alpha_hangul_ratio": filter_cfg.min_alpha_hangul_ratio,
                "max_whitespace_ratio": filter_cfg.max_whitespace_ratio,
                "max_punct_ratio": filter_cfg.max_punct_ratio,
                "language_policy": language,
            },
            "filter_reasons": dict(filter_stats.reasons),
            "filtered_documents": filtered_total,
            "duplicates_removed": deduper.stats.duplicates,
            "kept_documents": kept_docs,
            "kept_characters": kept_chars,
            "keep_rate": keep_rate,
            "hangul_document_ratio": (
                hangul_docs / kept_docs if kept_docs else 0.0
            ),
        },
        "wikipedia_sanity": {
            "documents_containing_hangul": hangul_docs,
            "documents_containing_latin": latin_docs,
            "documents_containing_digits": digit_docs,
            "documents_containing_math_like": math_docs,
            "note": "Counts only; not used for filtering.",
        },
        "long_documents": {
            "character_median": long_char_stats.percentile(50),
            "character_p50": long_char_stats.percentile(50),
            "character_p90": long_char_stats.percentile(90),
            "character_p95": long_char_stats.percentile(95),
            "character_p99": long_char_stats.percentile(99),
            "character_maximum": long_char_stats.max_v,
            "approx_docs_ge_10k_chars": long_10k_est,
            "approx_docs_ge_50k_chars": long_50k_est,
            "token_p50_sample": med_tok,
            "token_p90_sample": p90_tok,
            "token_p95_sample": p95_tok,
            "token_p99_sample": p99_tok,
            "token_maximum_sample": max_tok,
        },
        "token_sample": {
            "sample_documents": sample_documents_out,
            "sample_characters": sample_chars_out,
            "sample_tokens": sample_tokens_out,
            "sample_seed": seed,
            "average_tokens_per_document": avg_tok,
            "median_tokens_per_document": med_tok,
            "min_tokens_per_document": min_tok,
            "max_tokens_per_document": max_tok,
            "p90": p90_tok,
            "p95": p95_tok,
            "p99": p99_tok,
            "chars_per_token": chars_per_token_out,
            "eos_included": True,
        },
        "estimate": estimate_block,
        "tokenizer": {
            "path": str(tokenizer_path.as_posix()),
            "sha256": tok_fp,
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "invalid_token_ids": 0,
            "texts_encoded": encode_calls,
        },
        "integrity": {
            "hangul_preserved_check": hangul_docs > 0 if kept_docs else True,
            "original_files_modified": False,
            "full_tokenization_performed": do_full,
            "elapsed_seconds": elapsed,
        },
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_report(report)
    print(f"Wrote {output_path.resolve()}", flush=True)
    return report


def _print_report(r: dict[str, Any]) -> None:
    inp = r["input"]
    cl = r["cleaning"]
    ts = r["token_sample"]
    est = r["estimate"]
    ld = r["long_documents"]
    tok = r["tokenizer"]
    er = est["estimated_token_range"]
    source = r.get("source", "")
    is_edu = "edu" in source.lower() or r.get("education_score_statistics")

    title = (
        "FineWeb-Edu English Fast Audit"
        if is_edu
        else "FineWiki Korean Fast Audit"
        if "wiki" in source.lower()
        else f"Corpus Fast Audit ({source})"
    )

    print()
    print(title)
    print()
    print("Input")
    print("-----")
    print(f"Files: {inp['file_count']} ({', '.join(inp['files'])})")
    print(f"Disk size: {inp['disk_mb']:.2f} MB")
    print(f"Raw documents: {inp['raw_documents']:,}")
    print(f"Raw characters: {inp['raw_characters']:,}")
    print()
    print("Cleaning")
    print("--------")
    print(f"Kept: {cl['kept_documents']:,}")
    print(f"Filtered: {cl['filtered_documents']:,}")
    print(f"Duplicates: {cl['duplicates_removed']:,}")
    print(f"Keep rate: {cl['keep_rate']:.4f}")
    print()
    print("Filter Reasons")
    print("--------------")
    reasons = cl.get("filter_reasons") or {}
    if reasons:
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:,}")
    else:
        print("(none)")
    print()

    lang = r.get("language_statistics")
    if lang:
        ls = lang.get("language_score") or {}
        print("Language")
        print("--------")
        print(f"English ratio: {lang.get('english_ratio', 0):.4f}")
        mean = ls.get("mean", float("nan"))
        med = ls.get("median", float("nan"))
        p10 = ls.get("p10", float("nan"))
        mn = ls.get("minimum")
        print(f"Language score mean: {mean:.6f}" if isinstance(mean, float) else f"Language score mean: {mean}")
        print(f"Median: {med:.6f}" if isinstance(med, float) else f"Median: {med}")
        print(f"P10: {p10:.6f}" if isinstance(p10, float) else f"P10: {p10}")
        print(f"Minimum: {mn}")
        print()

    edu = r.get("education_score_statistics")
    if edu:
        print("Education Quality")
        print("-----------------")
        by = edu.get("by_int_score") or {}
        for k in sorted(by.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 0):
            entry = by[k]
            print(
                f"int_score {k}: {entry['documents']:,} "
                f"({entry['ratio']:.4f})"
            )
        print(f"Score >= 4 ratio: {edu.get('score_ge_4_ratio', 0):.4f}")
        print(f"Score 5 ratio: {edu.get('int_score_5_ratio', 0):.4f}")
        print()

    print("Token Sample")
    print("------------")
    print(f"Sample documents: {ts['sample_documents']:,}")
    print(f"Sample characters: {ts['sample_characters']:,}")
    print(f"Sample Siat tokens: {ts['sample_tokens']:,}")
    print(f"Average tokens/document: {ts['average_tokens_per_document']:.1f}")
    print(f"Median: {ts['median_tokens_per_document']:.1f}")
    print(f"P90: {ts['p90']:.1f}")
    print(f"P95: {ts['p95']:.1f}")
    print(f"P99: {ts['p99']:.1f}")
    print(f"Maximum: {ts['max_tokens_per_document']}")
    cpt = ts["chars_per_token"]
    print(
        f"Chars/token: {cpt:.4f}"
        if isinstance(cpt, float) and math.isfinite(cpt)
        else f"Chars/token: {cpt}"
    )
    print()
    print("Estimate")
    print("--------")
    print(
        f"Estimated total Siat tokens: ~{est['estimated_total_tokens']:,}"
        if est.get("is_estimate", True)
        else f"Total Siat tokens (exact): {est['estimated_total_tokens']:,}"
    )
    print(f"Estimated range: ~{er[0]:,}-{er[1]:,}")
    print(f"Method: {est['method']} (is_estimate={est['is_estimate']})")
    print()

    dtc = r.get("dataset_token_comparison")
    if dtc:
        print("Dataset Token Comparison")
        print("------------------------")
        print(f"Dataset token_count/sample: {dtc['dataset_tokens_per_document']:.2f}")
        print(f"Siat token count/sample: {dtc['siat_tokens_per_document']:.2f}")
        ratio = dtc.get("approx_ratio_siat_over_dataset", float("nan"))
        print(
            f"Ratio: {ratio:.4f}"
            if isinstance(ratio, float) and math.isfinite(ratio)
            else f"Ratio: {ratio}"
        )
        print()

    print("Long Documents")
    print("--------------")
    print(f"Character P95: {ld['character_p95']:.1f}")
    print(f"Character P99: {ld['character_p99']:.1f}")
    print(f"Character max: {ld['character_maximum']}")
    print(f"Token P95: {ld['token_p95_sample']:.1f}")
    print(f"Token P99: {ld['token_p99_sample']:.1f}")
    print(f"Token max: {ld['token_maximum_sample']}")
    print()
    print("Integrity")
    print("---------")
    print(f"Invalid token IDs: {tok['invalid_token_ids']}")
    print(f"Original files modified: {r['integrity']['original_files_modified']}")
    print()
    print("Performance")
    print("-----------")
    print(f"Elapsed: {r['integrity']['elapsed_seconds']:.1f}s")
    print(
        f"Full tokenization performed: "
        f"{r['integrity']['full_tokenization_performed']}"
    )
    print()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Siat Fast Corpus Audit")
    p.add_argument("--input", default="data/raw/finewiki_ko")
    p.add_argument("--format", default="parquet", choices=["parquet", "jsonl", "txt"])
    p.add_argument("--source", default="finewiki_ko")
    p.add_argument("--language", default="ko")
    p.add_argument("--tokenizer", default="tokenizer/siat-tokenizer.json")
    p.add_argument("--output", default="data/audits/finewiki_ko_audit.json")
    p.add_argument("--text-field", default="text")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--min-chars", type=int, default=32)
    p.add_argument("--sample-documents", type=int, default=DEFAULT_SAMPLE_DOCUMENTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Sample-tokenize mode (default). Overridden by --full-token-audit.",
    )
    p.add_argument(
        "--full-token-audit",
        action="store_true",
        default=False,
        help="Tokenize all kept documents (slow). Opt-in only.",
    )
    p.add_argument("--progress-every", type=int, default=50_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_corpus_audit(
        input_path=args.input,
        fmt=args.format,
        source=args.source,
        language=args.language,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        text_field=args.text_field,
        batch_size=args.batch_size,
        min_chars=args.min_chars,
        sample_documents=args.sample_documents,
        seed=args.seed,
        fast=args.fast,
        full_token_audit=args.full_token_audit,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
