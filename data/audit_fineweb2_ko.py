"""Audit FineWeb-2 Korean Parquet through the Siat data pipeline.

Measures raw / cleaned / deduped document stats and Siat tokenizer token
counts (with EOS). Estimates additional data needed for ~36.84M-param
Chinchilla-style training (~20 tokens/param). Does **not** write train.bin
or download extra shards.

Example::

    python -m data.audit_fineweb2_ko \\
        --input data/raw/fineweb2_ko \\
        --tokenizer tokenizer/siat-tokenizer.json \\
        --output data/audits/fineweb2_ko_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.dedup import ExactDeduper
from data.document import iter_parquet_documents, list_input_files, make_document_id
from data.filters import FilterConfig, FilterStats, filter_document
from data.build_pretraining_data import PIPELINE_VERSION
from tokenizer import load_tokenizer

SOURCE_NAME = "fineweb2_ko"
LANGUAGE = "ko"
EOS_TOKEN = "<|eos|>"
SIAT_PARAMS = 36_837_888
TOKENS_PER_PARAM = 20  # Chinchilla-style heuristic
ESTIMATED_TOKENS_NEEDED = SIAT_PARAMS * TOKENS_PER_PARAM

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_parquet_schema(path: Path, text_field: str = "text") -> list[str]:
    import pyarrow.parquet as pq

    files = list_input_files(path, ".parquet")
    names = list(pq.ParquetFile(files[0]).schema_arrow.names)
    if text_field not in names:
        raise ValueError(
            f"Expected text column {text_field!r} missing. Schema columns: {names}"
        )
    return names


def run_fineweb2_ko_audit(
    *,
    input_path: str | Path = "data/raw/fineweb2_ko",
    tokenizer_path: str | Path = "tokenizer/siat-tokenizer.json",
    output_path: str | Path = "data/audits/fineweb2_ko_audit.json",
    text_field: str = "text",
    batch_size: int = 1024,
    min_chars: int = 32,
    sample_count: int = 0,
    progress_every: int = 50_000,
    max_documents: int | None = None,
) -> dict[str, Any]:
    """Stream FineWeb-2 KO through Siat clean/filter/dedup/tokenize audit."""
    input_path = Path(input_path)
    tokenizer_path = Path(tokenizer_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Siat tokenizer not found: {tokenizer_path.resolve()}. "
            "Train or provide tokenizer/siat-tokenizer.json before audit."
        )

    files = list_input_files(input_path, ".parquet")
    disk_bytes = sum(f.stat().st_size for f in files)
    schema_cols = inspect_parquet_schema(input_path, text_field=text_field)
    print(f"Parquet files: {len(files)}", flush=True)
    print(f"Schema columns: {schema_cols}", flush=True)
    print(f"Using text field: {text_field!r}", flush=True)
    print(f"Disk size: {disk_bytes / 1e6:.2f} MB", flush=True)

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer missing {EOS_TOKEN!r}")
    tok_fp = sha256_file(tokenizer_path)

    filter_cfg = FilterConfig(min_chars=min_chars)
    filter_stats = FilterStats()
    deduper = ExactDeduper()

    raw_docs = 0
    raw_chars = 0
    parse_errors = 0
    replacement_chars = 0
    hangul_docs = 0
    kept_after_filter = 0
    token_total = 0
    eos_count = 0
    char_stats = LengthStats()
    tok_stats = LengthStats()
    samples: list[dict[str, Any]] = []
    sample_rng = random.Random(42)

    if eos_id < 0 or eos_id >= vocab_size:
        raise RuntimeError(f"Invalid EOS id {eos_id}")

    encode_batch_size = 256
    pending_texts: list[str] = []
    pending_ids: list[str] = []

    def flush_encode() -> None:
        nonlocal token_total, eos_count
        if not pending_texts:
            return
        if hasattr(tokenizer, "encode_batch"):
            encodings = tokenizer.encode_batch(pending_texts)
        else:
            encodings = [tokenizer.encode(t) for t in pending_texts]
        for i, enc in enumerate(encodings):
            ids = enc.ids
            doc_id = pending_ids[i]
            text = pending_texts[i]
            if ids and (min(ids) < 0 or max(ids) >= vocab_size):
                bad = next(tid for tid in ids if tid < 0 or tid >= vocab_size)
                raise RuntimeError(
                    f"Invalid token id {bad} (vocab_size={vocab_size}) "
                    f"doc_id={doc_id}"
                )
            n_tok = len(ids) + 1  # EOS
            token_total += n_tok
            eos_count += 1
            tok_stats.add(n_tok)
            if sample_count > 0:
                item = {
                    "document_id": doc_id,
                    "chars": len(text),
                    "tokens": n_tok,
                    "preview": text[:120].replace("\n", " "),
                }
                if len(samples) < sample_count:
                    samples.append(item)
                else:
                    j = sample_rng.randint(0, kept_after_filter - 1)
                    if j < sample_count:
                        samples[j] = item
        pending_texts.clear()
        pending_ids.clear()

    t0 = time.perf_counter()
    print("Auditing (streaming)…", flush=True)

    for doc, err in iter_parquet_documents(
        input_path,
        source=SOURCE_NAME,
        text_field=text_field,
        language=LANGUAGE,
        batch_size=batch_size,
    ):
        if err is not None:
            parse_errors += 1
            continue
        assert doc is not None
        raw_docs += 1
        raw_chars += len(doc.text)
        replacement_chars += doc.text.count("\ufffd")
        char_stats.add(len(doc.text))

        kept, reason = filter_document(doc, filter_cfg, filter_stats)
        if kept is None:
            if max_documents is not None and raw_docs >= max_documents:
                break
            continue

        kept.document_id = make_document_id(kept.source, kept.text)
        unique = deduper.consider(kept)
        if unique is None:
            if max_documents is not None and raw_docs >= max_documents:
                break
            continue

        kept_after_filter += 1
        if _has_hangul(unique.text):
            hangul_docs += 1

        pending_texts.append(unique.text)
        pending_ids.append(unique.document_id)
        if len(pending_texts) >= encode_batch_size:
            flush_encode()

        if progress_every > 0 and raw_docs % progress_every == 0:
            flush_encode()
            elapsed = time.perf_counter() - t0
            print(
                f"  raw={raw_docs:,} kept={kept_after_filter:,} "
                f"dup={deduper.stats.duplicates:,} "
                f"tokens={token_total:,} ({elapsed:.1f}s)",
                flush=True,
            )

        if max_documents is not None and raw_docs >= max_documents:
            break

    flush_encode()
    elapsed = time.perf_counter() - t0
    disk_mb = disk_bytes / 1e6
    tokens_per_mb = token_total / disk_mb if disk_mb > 0 else 0.0
    missing = max(0, ESTIMATED_TOKENS_NEEDED - token_total)
    additional_mb = (
        missing / tokens_per_mb if tokens_per_mb > 0 else float("inf")
    )
    shard_mb = disk_mb / max(len(files), 1)
    additional_shards = (
        additional_mb / shard_mb if shard_mb > 0 and math.isfinite(additional_mb) else float("inf")
    )

    filtered_total = sum(filter_stats.reasons.values())
    keep_rate = kept_after_filter / raw_docs if raw_docs else 0.0

    report: dict[str, Any] = {
        "source": SOURCE_NAME,
        "language": LANGUAGE,
        "pipeline_version": PIPELINE_VERSION,
        "unicode_normalization": "NFC",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path.as_posix()),
            "files": [f.name for f in files],
            "file_count": len(files),
            "disk_bytes": disk_bytes,
            "disk_mb": disk_mb,
            "schema_columns": schema_cols,
            "text_field": text_field,
            "raw_documents": raw_docs,
            "raw_characters": raw_chars,
            "average_chars_per_document": char_stats.mean(),
            "parse_errors": parse_errors,
        },
        "cleaning": {
            "filter_config": {
                "min_chars": filter_cfg.min_chars,
                "max_repeat_ratio": filter_cfg.max_repeat_ratio,
                "min_alpha_hangul_ratio": filter_cfg.min_alpha_hangul_ratio,
                "max_whitespace_ratio": filter_cfg.max_whitespace_ratio,
                "max_punct_ratio": filter_cfg.max_punct_ratio,
            },
            "filter_reasons": dict(filter_stats.reasons),
            "filtered_documents": filtered_total,
            "duplicates_removed": deduper.stats.duplicates,
            "kept_documents": kept_after_filter,
            "keep_rate": keep_rate,
            "hangul_document_ratio": (
                hangul_docs / kept_after_filter if kept_after_filter else 0.0
            ),
            "replacement_char_count_in_raw": replacement_chars,
        },
        "tokenizer": {
            "path": str(tokenizer_path.as_posix()),
            "sha256": tok_fp,
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "total_siat_tokens": token_total,
            "eos_count": eos_count,
            "average_tokens_per_document": tok_stats.mean(),
            "median_tokens_per_document": tok_stats.percentile(50),
            "min_tokens_per_document": tok_stats.min_v,
            "max_tokens_per_document": tok_stats.max_v,
            "p90": tok_stats.percentile(90),
            "p95": tok_stats.percentile(95),
            "p99": tok_stats.percentile(99),
            "invalid_token_ids": 0,
        },
        "scale_estimate": {
            "model_params": SIAT_PARAMS,
            "rule_of_thumb": f"~{TOKENS_PER_PARAM} tokens/param (Chinchilla-style heuristic)",
            "estimated_tokens_needed": ESTIMATED_TOKENS_NEEDED,
            "available_tokens": token_total,
            "coverage_vs_estimate": (
                token_total / ESTIMATED_TOKENS_NEEDED
                if ESTIMATED_TOKENS_NEEDED
                else None
            ),
            "tokens_missing_vs_estimate": missing,
            "tokens_per_mb_observed": tokens_per_mb,
            "additional_mb_estimate": additional_mb,
            "additional_shards_estimate": additional_shards,
            "note": (
                "estimated_tokens_needed is a heuristic (~20 tok/param), "
                "not a guarantee of optimal training quality."
            ),
        },
        "integrity": {
            "hangul_preserved_check": hangul_docs > 0,
            "original_files_modified": False,
            "elapsed_seconds": elapsed,
        },
        "samples": samples,
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_report(report)
    print(f"Wrote {output_path.resolve()}")
    return report


def _print_report(r: dict[str, Any]) -> None:
    inp = r["input"]
    cl = r["cleaning"]
    tok = r["tokenizer"]
    sc = r["scale_estimate"]
    print()
    print("FineWeb-2 Korean Audit")
    print()
    print("Input")
    print("-----")
    print(f"Files: {inp['file_count']} ({', '.join(inp['files'])})")
    print(f"Disk size: {inp['disk_mb']:.2f} MB")
    print(f"Raw documents: {inp['raw_documents']:,}")
    print(f"Raw characters: {inp['raw_characters']:,}")
    print(f"Average characters/document: {inp['average_chars_per_document']:.1f}")
    print()
    print("Cleaning")
    print("--------")
    print(f"Kept: {cl['kept_documents']:,}")
    print(f"Filtered: {cl['filtered_documents']:,}")
    print(f"Duplicates: {cl['duplicates_removed']:,}")
    print(f"Keep rate: {cl['keep_rate']:.4f}")
    print(f"Filter reasons: {cl['filter_reasons']}")
    print(f"Hangul doc ratio (kept): {cl['hangul_document_ratio']:.4f}")
    print()
    print("Tokenizer")
    print("---------")
    print(f"Tokenizer: {tok['path']}")
    print(f"Vocab size: {tok['vocab_size']}")
    print(f"Total Siat tokens: {tok['total_siat_tokens']:,}")
    print(f"Average tokens/document: {tok['average_tokens_per_document']:.1f}")
    print(f"Median tokens/document: {tok['median_tokens_per_document']:.1f}")
    print(f"Min/Max: {tok['min_tokens_per_document']} / {tok['max_tokens_per_document']}")
    print(f"P90: {tok['p90']:.1f}")
    print(f"P95: {tok['p95']:.1f}")
    print(f"P99: {tok['p99']:.1f}")
    print()
    print("Scale estimate (heuristic)")
    print("--------------------------")
    print(f"Model params: {sc['model_params']:,}")
    print(f"Rule of thumb: {sc['rule_of_thumb']}")
    print(f"Estimated tokens needed: {sc['estimated_tokens_needed']:,}")
    print(f"Available (this data after pipeline): {sc['available_tokens']:,}")
    cov = sc["coverage_vs_estimate"]
    print(f"Coverage vs estimate: {cov:.4f}" if cov is not None else "Coverage: n/a")
    print(f"Missing vs estimate: {sc['tokens_missing_vs_estimate']:,}")
    print(f"Tokens per MB (observed): {sc['tokens_per_mb_observed']:.1f}")
    print(f"Additional MB (estimate): {sc['additional_mb_estimate']:.1f}")
    print(f"Additional shards (estimate): {sc['additional_shards_estimate']:.2f}")
    print(f"Note: {sc['note']}")
    print()
    print("Integrity")
    print("---------")
    print(f"Hangul preserved (any kept docs): {r['integrity']['hangul_preserved_check']}")
    print(f"Invalid token IDs: {tok['invalid_token_ids']}")
    print(f"Original files modified: {r['integrity']['original_files_modified']}")
    print(f"Elapsed: {r['integrity']['elapsed_seconds']:.1f}s")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit FineWeb-2 Korean for Siat tokens")
    p.add_argument("--input", default="data/raw/fineweb2_ko")
    p.add_argument("--tokenizer", default="tokenizer/siat-tokenizer.json")
    p.add_argument("--output", default="data/audits/fineweb2_ko_audit.json")
    p.add_argument("--text-field", default="text")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--min-chars", type=int, default=32)
    p.add_argument("--sample-count", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50_000)
    p.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap for smoke/debug (omit for full shard)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_fineweb2_ko_audit(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        text_field=args.text_field,
        batch_size=args.batch_size,
        min_chars=args.min_chars,
        sample_count=args.sample_count,
        progress_every=args.progress_every,
        max_documents=args.max_documents,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
