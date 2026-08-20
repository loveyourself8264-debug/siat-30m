"""Build Siat pretraining binary datasets from a multi-source manifest.

Pipeline:
  Raw corpora → NFC clean → quality filter → global exact dedup
  → document-level train/val split → char-proxy train mixing
  → Siat tokenizer + EOS → train.bin / val.bin + metadata / statistics

No network downloads. Streaming-friendly (does not hold the full token stream).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from config import ModelConfig
from data.cleaning import normalize_text
from data.dedup import ExactDeduper, sort_key
from data.document import (
    Document,
    iter_source_documents,
    load_manifest,
    make_document_id,
)
from data.filters import FilterConfig, FilterStats, filter_config_from_dict, filter_document
from data.mixing import mix_train_documents, split_documents
from data.preprocess import WRITE_BUFFER_TOKENS, choose_token_dtype
from tokenizer import load_tokenizer

PIPELINE_VERSION = "siat-data-v1"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
DEFAULT_VOCAB_SIZE = ModelConfig.siat_30m().vocab_size
PROGRESS_EVERY = 500


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_tokenizer(tokenizer: Tokenizer, expected_vocab: int) -> int:
    vocab = tokenizer.get_vocab_size()
    if vocab != expected_vocab:
        raise ValueError(
            f"Tokenizer vocab_size={vocab} does not match expected "
            f"ModelConfig/CLI vocab_size={expected_vocab}."
        )
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer missing required token {EOS_TOKEN!r}.")
    if unk_id is None:
        raise ValueError(f"Tokenizer missing required token {UNK_TOKEN!r}.")
    return eos_id


class TokenBinWriter:
    """Append token ids to a raw binary file with a fixed-size buffer."""

    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        self.buffer = np.empty(WRITE_BUFFER_TOKENS, dtype=self.dtype)
        self.buf_len = 0
        self.total_tokens = 0
        self.eos_count = 0
        self._handle = self.path.open("wb")
        self._max_id = int(np.iinfo(self.dtype).max)

    def write_ids(self, ids: list[int], *, eos_id: int) -> int:
        seq = list(ids) + [eos_id]
        for token_id in seq:
            if token_id < 0 or token_id > self._max_id:
                raise ValueError(
                    f"Token id {token_id} out of range for dtype {self.dtype}."
                )
            self.buffer[self.buf_len] = token_id
            self.buf_len += 1
            if self.buf_len == WRITE_BUFFER_TOKENS:
                self._flush()
        self.eos_count += 1
        return len(seq)

    def _flush(self) -> None:
        if self.buf_len == 0:
            return
        self._handle.write(self.buffer[: self.buf_len].tobytes())
        self.total_tokens += self.buf_len
        self.buf_len = 0

    def close(self) -> int:
        self._flush()
        self._handle.close()
        return self.total_tokens


def collect_documents(
    manifest: dict[str, Any],
    *,
    filter_cfg: FilterConfig,
    max_documents: int | None = None,
    text_field_override: str | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    """Parse → clean → filter all sources; return cleaned docs + ingest stats."""
    filter_stats = FilterStats()
    ingest: dict[str, Any] = {
        "input_documents": 0,
        "parse_errors": 0,
        "parse_error_reasons": defaultdict(int),
        "by_source": {},
        "filter_reasons": filter_stats.reasons,
        "filter_samples": filter_stats.samples,
    }

    cleaned: list[Document] = []
    for src in manifest["sources"]:
        name = src["name"]
        path = src["path"]
        fmt = src["format"]
        language = src.get("language")
        text_field = text_field_override or src.get("text_field", "text")
        src_stats = {
            "input": 0,
            "parse_errors": 0,
            "kept_after_filter": 0,
            "filtered": 0,
            "license": src.get("license"),
            "url": src.get("url"),
            "notes": src.get("notes"),
            "language": language,
            "weight": float(src["weight"]),
            "path": path,
            "format": fmt,
        }
        ingest["by_source"][name] = src_stats

        for doc, err in iter_source_documents(
            path,
            source=name,
            fmt=fmt,
            text_field=text_field,
            language=language,
        ):
            if err is not None:
                ingest["parse_errors"] += 1
                ingest["parse_error_reasons"][err.split(":")[0]] += 1
                src_stats["parse_errors"] += 1
                warnings.warn(f"Skipping malformed input: {err}", UserWarning)
                continue

            assert doc is not None
            ingest["input_documents"] += 1
            src_stats["input"] += 1

            kept, reason = filter_document(doc, filter_cfg, filter_stats)
            if kept is None:
                src_stats["filtered"] += 1
                continue

            kept.document_id = make_document_id(kept.source, kept.text)
            cleaned.append(kept)
            src_stats["kept_after_filter"] += 1

            if max_documents is not None and ingest["input_documents"] >= max_documents:
                break
        if max_documents is not None and ingest["input_documents"] >= max_documents:
            break

    ingest["parse_error_reasons"] = dict(ingest["parse_error_reasons"])
    ingest["kept_after_filter"] = len(cleaned)
    ingest["filtered_documents"] = sum(
        s["filtered"] for s in ingest["by_source"].values()
    )
    return cleaned, ingest


def dedup_documents(docs: list[Document]) -> tuple[list[Document], ExactDeduper]:
    """Global exact dedup with deterministic winner order."""
    ordered = sorted(docs, key=sort_key)
    deduper = ExactDeduper()
    kept: list[Document] = []
    for doc in ordered:
        out = deduper.consider(doc)
        if out is not None:
            kept.append(out)
    return kept, deduper


def write_docs_tokens(
    docs: list[Document],
    tokenizer: Tokenizer,
    eos_id: int,
    out_bin: Path,
    dtype: np.dtype,
) -> tuple[int, int, dict[str, int]]:
    """Tokenize documents streaming to ``out_bin``. Returns tokens, eos, per-source tokens."""
    writer = TokenBinWriter(out_bin, dtype)
    per_source: dict[str, int] = defaultdict(int)
    try:
        for i, doc in enumerate(docs, start=1):
            ids = tokenizer.encode(doc.text).ids
            # Sanity: ids in range
            for tid in ids:
                if tid < 0 or tid >= tokenizer.get_vocab_size():
                    raise ValueError(
                        f"Token id {tid} outside vocab [0, {tokenizer.get_vocab_size()})."
                    )
            n = writer.write_ids(ids, eos_id=eos_id)
            per_source[doc.source] += n
            if i % PROGRESS_EVERY == 0:
                print(
                    f"  tokenized {i}/{len(docs)} docs | "
                    f"tokens_written≈{writer.total_tokens + writer.buf_len}"
                )
    finally:
        total = writer.close()
    return total, writer.eos_count, dict(per_source)


def build_pretraining_data(
    *,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    validation_ratio: float = 0.01,
    seed: int = 42,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_chars: int = 32,
    max_documents: int | None = None,
    dry_run: bool = False,
    max_oversample: int = 3,
    text_field: str | None = None,
    filter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline; return metadata dict (also written to disk unless dry_run skips bins)."""
    manifest_path = Path(manifest_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    # Resolve relative source paths: try manifest directory, then cwd (repo root).
    for src in manifest["sources"]:
        p = Path(src["path"])
        if p.is_absolute():
            continue
        candidate = (manifest_path.parent / p).resolve()
        if not candidate.exists():
            candidate = (Path.cwd() / p).resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Source path for {src['name']!r} not found: {src['path']!r} "
                f"(tried { (manifest_path.parent / p).resolve() } and "
                f"{(Path.cwd() / p).resolve() })."
            )
        src["path"] = str(candidate)

    filter_cfg = filter_config_from_dict(
        {
            "min_chars": min_chars,
            **(filter_overrides or {}),
        }
    )

    print(f"Pipeline {PIPELINE_VERSION}")
    print(f"Manifest: {manifest_path}")
    print(f"Tokenizer: {tokenizer_path}")

    cleaned, ingest = collect_documents(
        manifest,
        filter_cfg=filter_cfg,
        max_documents=max_documents,
        text_field_override=text_field,
    )
    print(
        f"Ingest: input={ingest['input_documents']} "
        f"kept_after_filter={ingest['kept_after_filter']} "
        f"filtered={ingest['filtered_documents']} "
        f"parse_errors={ingest['parse_errors']}"
    )

    deduped, deduper = dedup_documents(cleaned)
    print(
        f"Dedup: kept={deduper.stats.kept} "
        f"duplicates={deduper.stats.duplicates}"
    )

    split = split_documents(
        deduped, validation_ratio=validation_ratio, seed=seed
    )
    print(f"Split: train_docs={len(split.train)} val_docs={len(split.val)}")

    mixed_train, mix_stats = mix_train_documents(
        split.train, manifest["sources"], max_oversample=max_oversample
    )
    print(f"Mix: train_stream_docs={len(mixed_train)} (char-proxy weights)")

    if dry_run:
        result = {
            "dry_run": True,
            "pipeline_version": PIPELINE_VERSION,
            "ingest": ingest,
            "dedup": {
                "kept": deduper.stats.kept,
                "duplicates": deduper.stats.duplicates,
                "by_source": deduper.stats.by_source,
            },
            "split": {
                "train_documents": len(split.train),
                "val_documents": len(split.val),
            },
            "mix": mix_stats.by_source,
            "avg_chars_kept": (
                sum(len(d.text) for d in deduped) / max(len(deduped), 1)
            ),
            "filter_reasons": dict(ingest["filter_reasons"]),
        }
        stats_path = output_dir / "statistics.json"
        stats_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Dry-run statistics → {stats_path}")
        return result

    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = validate_tokenizer(tokenizer, vocab_size)
    dtype = choose_token_dtype(vocab_size)
    tok_fp = sha256_file(tokenizer_path)
    man_fp = sha256_file(manifest_path)

    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"

    print("Tokenizing train…")
    train_tokens, train_eos, train_src_tok = write_docs_tokens(
        mixed_train, tokenizer, eos_id, train_bin, dtype
    )
    print("Tokenizing val…")
    val_tokens, val_eos, val_src_tok = write_docs_tokens(
        split.val, tokenizer, eos_id, val_bin, dtype
    )

    total_tok = train_tokens + val_tokens
    actual_ratios: dict[str, float] = {}
    for name in {s["name"] for s in manifest["sources"]}:
        t = train_src_tok.get(name, 0)
        actual_ratios[name] = (t / train_tokens) if train_tokens else 0.0

    # Per-source document counts after mix / val
    train_docs_by_src: dict[str, int] = defaultdict(int)
    for d in mixed_train:
        train_docs_by_src[d.source] += 1
    val_docs_by_src: dict[str, int] = defaultdict(int)
    for d in split.val:
        val_docs_by_src[d.source] += 1

    weights = {
        s["name"]: float(s["weight"]) for s in manifest["sources"]
    }
    wsum = sum(weights.values()) or 1.0
    target_weights = {k: v / wsum for k, v in weights.items()}

    sources_out = []
    for src in manifest["sources"]:
        name = src["name"]
        sources_out.append(
            {
                "name": name,
                "path": src["path"],
                "format": src["format"],
                "language": src.get("language"),
                "license": src.get("license"),
                "url": src.get("url"),
                "notes": src.get("notes"),
                "weight": float(src["weight"]),
                "target_weight": target_weights[name],
                "actual_train_token_ratio": actual_ratios.get(name, 0.0),
                "input_docs": ingest["by_source"][name]["input"],
                "kept_after_filter": ingest["by_source"][name]["kept_after_filter"],
                "filtered": ingest["by_source"][name]["filtered"],
                "duplicates": deduper.stats.by_source.get(name, {}).get(
                    "duplicates", 0
                ),
                "train_docs_in_stream": train_docs_by_src.get(name, 0),
                "val_docs": val_docs_by_src.get(name, 0),
                "train_tokens": train_src_tok.get(name, 0),
                "val_tokens": val_src_tok.get(name, 0),
                "mix": mix_stats.by_source.get(name, {}),
            }
        )

    kept_docs = len(deduped)
    avg_tok = (
        (train_tokens + val_tokens) / max(len(mixed_train) + len(split.val), 1)
    )

    metadata: dict[str, Any] = {
        "tokenizer": str(tokenizer_path.as_posix()),
        "tokenizer_sha256": tok_fp,
        "manifest": str(manifest_path.as_posix()),
        "manifest_sha256": man_fp,
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_documents": len(mixed_train),
        "val_documents": len(split.val),
        "unique_documents_after_dedup": kept_docs,
        "eos_token_id": eos_id,
        "eos_count_train": train_eos,
        "eos_count_val": val_eos,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "pipeline_version": PIPELINE_VERSION,
        "train_bin": train_bin.name,
        "val_bin": val_bin.name if split.val else None,
        "num_train_documents": len(mixed_train),
        "num_val_documents": len(split.val),
        "average_tokens_per_document": avg_tok,
        "fingerprint": hashlib.sha256(
            f"{tok_fp}:{man_fp}:{PIPELINE_VERSION}:{seed}:{validation_ratio}".encode()
        ).hexdigest(),
        "min_chars": min_chars,
        "max_oversample": max_oversample,
        "mixing": "char_proxy_target_token_budget",
        "unicode_normalization": "NFC",
    }

    statistics: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "input_documents": ingest["input_documents"],
        "kept_after_filter": ingest["kept_after_filter"],
        "filtered_documents": ingest["filtered_documents"],
        "filter_reasons": dict(ingest["filter_reasons"]),
        "filter_samples": ingest["filter_samples"],
        "parse_errors": ingest["parse_errors"],
        "duplicates_removed": deduper.stats.duplicates,
        "documents_after_dedup": deduper.stats.kept,
        "train_documents": len(mixed_train),
        "val_documents": len(split.val),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "average_tokens_per_document": avg_tok,
        "eos_count": train_eos + val_eos,
        "sources": sources_out,
        "mix": mix_stats.by_source,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sources.json").write_text(
        json.dumps(sources_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_cfg = {
        "validation_ratio": validation_ratio,
        "seed": seed,
        "min_chars": min_chars,
        "dedup": True,
        "pipeline_version": PIPELINE_VERSION,
        "max_oversample": max_oversample,
        "vocab_size": vocab_size,
    }
    (output_dir / "build_config.json").write_text(
        json.dumps(build_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("---------- Build summary ----------")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens:   {val_tokens:,}")
    print(f"Train docs:   {len(mixed_train)}")
    print(f"Val docs:     {len(split.val)}")
    print(f"Avg tokens/doc: {avg_tok:.1f}")
    print(f"EOS count: {train_eos + val_eos}")
    print(f"dtype: {dtype}")
    for row in sources_out:
        print(
            f"  {row['name']}: target={row['target_weight']:.3f} "
            f"actual={row['actual_train_token_ratio']:.3f} "
            f"train_tok={row['train_tokens']}"
        )
    print(f"Wrote {output_dir.resolve()}")
    return metadata


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Siat pretraining train/val token binaries from a manifest."
    )
    p.add_argument("--manifest", required=True, help="JSON manifest path")
    p.add_argument(
        "--tokenizer",
        default="tokenizer/siat-tokenizer.json",
        help="Siat tokenizer JSON",
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--validation-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--min-chars", type=int, default=32)
    p.add_argument("--max-documents", type=int, default=None)
    p.add_argument("--max-oversample", type=int, default=3)
    p.add_argument("--text-field", type=str, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run clean/filter/dedup/split/mix stats without writing bins",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    build_pretraining_data(
        manifest_path=args.manifest,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        vocab_size=args.vocab_size,
        min_chars=args.min_chars,
        max_documents=args.max_documents,
        dry_run=args.dry_run,
        max_oversample=args.max_oversample,
        text_field=args.text_field,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
