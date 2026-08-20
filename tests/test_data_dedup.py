"""Tests for exact document deduplication."""

from __future__ import annotations

from data.cleaning import normalize_text
from data.dedup import ExactDeduper, sort_key
from data.document import Document, make_document_id


def test_exact_duplicate_removal():
    text = normalize_text("동일한 본문입니다. " * 5)
    a = Document(text=text, source="a", document_id=make_document_id("a", text))
    b = Document(text=text, source="b", document_id=make_document_id("b", text))
    deduper = ExactDeduper()
    assert deduper.consider(a) is not None
    assert deduper.consider(b) is None
    assert deduper.stats.duplicates == 1
    assert deduper.stats.kept == 1


def test_different_documents_kept():
    t1 = normalize_text("첫 번째 서로 다른 문서입니다. " * 4)
    t2 = normalize_text("두 번째 서로 다른 문서입니다. " * 4)
    deduper = ExactDeduper()
    assert deduper.consider(Document(text=t1, source="s")) is not None
    assert deduper.consider(Document(text=t2, source="s")) is not None
    assert deduper.stats.kept == 2


def test_cross_source_deterministic_winner():
    text = normalize_text("cross source duplicate body text for tests. " * 3)
    docs = [
        Document(text=text, source="z_source", document_id=make_document_id("z_source", text)),
        Document(text=text, source="a_source", document_id=make_document_id("a_source", text)),
    ]
    ordered = sorted(docs, key=sort_key)
    assert ordered[0].source == "a_source"
    deduper = ExactDeduper()
    kept = [deduper.consider(d) for d in ordered]
    assert kept[0] is not None
    assert kept[1] is None


def test_document_id_deterministic():
    text = "stable id text"
    a = make_document_id("src", text)
    b = make_document_id("src", text)
    c = make_document_id("other", text)
    assert a == b
    assert a != c
    assert len(a) == 64
