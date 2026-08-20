"""Tests for train/val split and char-proxy mixing."""

from __future__ import annotations

from data.cleaning import normalize_text
from data.document import Document, make_document_id
from data.mixing import assign_split, mix_train_documents, split_documents


def _docs(n: int, source: str, prefix: str) -> list[Document]:
    out = []
    for i in range(n):
        text = normalize_text(f"{prefix} document number {i}. " + ("body " * 20))
        out.append(
            Document(
                text=text,
                source=source,
                document_id=make_document_id(source, text),
            )
        )
    return out


def test_same_seed_same_split():
    docs = _docs(20, "s", "alpha")
    a = split_documents(docs, validation_ratio=0.2, seed=42)
    b = split_documents(docs, validation_ratio=0.2, seed=42)
    assert {d.document_id for d in a.train} == {d.document_id for d in b.train}
    assert {d.document_id for d in a.val} == {d.document_id for d in b.val}


def test_no_train_val_overlap():
    docs = _docs(30, "s", "beta")
    split = split_documents(docs, validation_ratio=0.2, seed=7)
    train_ids = {d.document_id for d in split.train}
    val_ids = {d.document_id for d in split.val}
    assert not (train_ids & val_ids)
    assert len(split.val) >= 1
    assert len(split.train) >= 1


def test_assign_split_stable():
    text = normalize_text("stable split body " * 10)
    doc = Document(text=text, source="s", document_id=make_document_id("s", text))
    assert assign_split(doc, validation_ratio=0.2, seed=1) == assign_split(
        doc, validation_ratio=0.2, seed=1
    )


def test_mixing_weights_and_determinism():
    ko = _docs(10, "korean_general", "ko")
    en = _docs(10, "english_general", "en")
    sources = [
        {"name": "korean_general", "weight": 0.7},
        {"name": "english_general", "weight": 0.3},
    ]
    mixed_a, stats_a = mix_train_documents(ko + en, sources, max_oversample=2)
    mixed_b, _ = mix_train_documents(ko + en, sources, max_oversample=2)
    assert [d.document_id for d in mixed_a] == [d.document_id for d in mixed_b]
    assert "korean_general" in stats_a.by_source
    assert stats_a.by_source["korean_general"]["docs"] >= 1
    # Char proxy: Korean target higher → typically more chars allocated
    assert (
        stats_a.by_source["korean_general"]["actual_chars"]
        >= stats_a.by_source["english_general"]["actual_chars"]
    )
