"""Exact document deduplication (SHA256 of cleaned text)."""

from __future__ import annotations

from dataclasses import dataclass, field

from data.document import Document, content_hash, make_document_id


@dataclass
class DedupStats:
    seen: int = 0
    kept: int = 0
    duplicates: int = 0
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def _src(self, source: str) -> dict[str, int]:
        if source not in self.by_source:
            self.by_source[source] = {"seen": 0, "kept": 0, "duplicates": 0}
        return self.by_source[source]


class ExactDeduper:
    """Global exact-text deduper. First deterministic winner is kept.

    Callers should feed documents sorted by ``(source, document_id)`` for
    deterministic cross-source winners, or rely on insertion order if already
    sorted externally.
    """

    def __init__(self) -> None:
        self._hashes: set[str] = set()
        self.stats = DedupStats()

    def consider(self, doc: Document) -> Document | None:
        """Return ``doc`` if new, else ``None`` if duplicate."""
        text = doc.text
        h = content_hash(text)
        src = self.stats._src(doc.source)
        self.stats.seen += 1
        src["seen"] += 1

        if h in self._hashes:
            self.stats.duplicates += 1
            src["duplicates"] += 1
            return None

        self._hashes.add(h)
        if not doc.document_id:
            doc.document_id = make_document_id(doc.source, text)
        self.stats.kept += 1
        src["kept"] += 1
        return doc


def sort_key(doc: Document) -> tuple[str, str]:
    did = doc.document_id or make_document_id(doc.source, doc.text)
    return (doc.source, did)
