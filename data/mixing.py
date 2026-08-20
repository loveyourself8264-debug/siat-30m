"""Train/validation split and char-proxy source mixing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from data.document import Document, make_document_id


@dataclass
class SplitResult:
    train: list[Document]
    val: list[Document]


@dataclass
class MixStats:
    """Per-source mixing / oversample statistics."""

    by_source: dict[str, dict[str, float | int]] = field(default_factory=dict)


def assign_split(
    doc: Document,
    *,
    validation_ratio: float,
    seed: int = 42,
) -> str:
    """Return ``'val'`` or ``'train'`` using document-id hash (order-stable).

    ``seed`` mixes into the bucket so different seeds change the split while
    remaining deterministic for a fixed id.
    """
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio must be in [0.0, 1.0), got {validation_ratio}."
        )
    if validation_ratio == 0.0:
        return "train"
    did = doc.document_id or make_document_id(doc.source, doc.text)
    # Mix seed into hash domain without non-determinism.
    bucket_src = f"{seed}:{did}"
    # Use first 8 hex digits of SHA256 via document id when possible.
    # document_id is already sha256 hex; fold seed by XOR of ints.
    base = int(did[:8], 16)
    mixed = (base ^ (seed * 2654435761 & 0xFFFFFFFF)) % 10_000
    threshold = int(validation_ratio * 10_000)
    return "val" if mixed < threshold else "train"


def split_documents(
    docs: list[Document],
    *,
    validation_ratio: float,
    seed: int = 42,
) -> SplitResult:
    """Document-level split; ensures both sides non-empty when possible."""
    train: list[Document] = []
    val: list[Document] = []
    for doc in docs:
        if assign_split(doc, validation_ratio=validation_ratio, seed=seed) == "val":
            val.append(doc)
        else:
            train.append(doc)

    # Guarantee at least one train doc when we have >=1 doc.
    if not train and val:
        # Move the lexicographically first val doc to train.
        val_sorted = sorted(
            val,
            key=lambda d: (
                d.source,
                d.document_id or make_document_id(d.source, d.text),
            ),
        )
        moved = val_sorted[0]
        val = [d for d in val if d is not moved]
        train = [moved]
    if validation_ratio > 0 and not val and len(train) >= 2:
        train_sorted = sorted(
            train,
            key=lambda d: (
                d.source,
                d.document_id or make_document_id(d.source, d.text),
            ),
        )
        moved = train_sorted[0]
        train = [d for d in train if d is not moved]
        val = [moved]

    train_ids = {d.document_id for d in train}
    val_ids = {d.document_id for d in val}
    if train_ids & val_ids:
        raise RuntimeError("Train/validation document id overlap detected.")

    return SplitResult(train=train, val=val)


def _normalize_weights(sources: list[dict]) -> dict[str, float]:
    weights = {s["name"]: float(s["weight"]) for s in sources}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sum of source weights must be > 0.")
    return {k: v / total for k, v in weights.items()}


def mix_train_documents(
    train_docs: list[Document],
    sources: list[dict],
    *,
    max_oversample: int = 3,
) -> tuple[list[Document], MixStats]:
    """Assemble train stream targeting weight shares via char-length proxy.

    Documents within each source are sorted by ``document_id``. Greedy fill
    toward ``weight * total_chars``. If a source is short, oversample up to
    ``max_oversample`` full passes (recorded in stats).
    """
    weights = _normalize_weights(sources)
    by_source: dict[str, list[Document]] = defaultdict(list)
    for doc in train_docs:
        by_source[doc.source].append(doc)

    for name in by_source:
        by_source[name].sort(
            key=lambda d: d.document_id or make_document_id(d.source, d.text)
        )

    total_chars = sum(len(d.text) for d in train_docs) or 1
    stats = MixStats()
    mixed: list[Document] = []

    # Preserve deterministic interleaving by processing sources in name order,
    # appending selected docs; final stream is concatenation in source-name order.
    for name in sorted(weights.keys()):
        target_share = weights[name]
        target_chars = target_share * total_chars
        pool = by_source.get(name, [])
        selected: list[Document] = []
        used_chars = 0
        repeats = 0

        if not pool:
            stats.by_source[name] = {
                "target_weight": target_share,
                "target_chars": int(target_chars),
                "actual_chars": 0,
                "docs": 0,
                "repeat_passes": 0,
            }
            continue

        # First pass: take all unique docs up to target (or all if under).
        for doc in pool:
            selected.append(doc)
            used_chars += len(doc.text)
            if used_chars >= target_chars:
                break

        # Oversample if still under target.
        pass_idx = 1
        while used_chars < target_chars and pass_idx < max_oversample:
            for doc in pool:
                selected.append(doc)
                used_chars += len(doc.text)
                if used_chars >= target_chars:
                    break
            pass_idx += 1
            repeats = pass_idx - 1

        mixed.extend(selected)
        stats.by_source[name] = {
            "target_weight": target_share,
            "target_chars": int(target_chars),
            "actual_chars": used_chars,
            "docs": len(selected),
            "unique_docs": len(pool),
            "repeat_passes": repeats,
        }

    # Actual char ratios after mix
    mix_total = sum(len(d.text) for d in mixed) or 1
    for name, row in stats.by_source.items():
        row["actual_char_ratio"] = float(row["actual_chars"]) / mix_total

    return mixed, stats
