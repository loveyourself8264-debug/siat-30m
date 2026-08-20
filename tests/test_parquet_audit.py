"""Tests for parquet ingest and FineWeb-2 KO audit (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.audit_fineweb2_ko import run_fineweb2_ko_audit
from data.cleaning import normalize_text
from data.document import iter_parquet_documents
from tokenizer.train_tokenizer import train_tokenizer


def _write_parquet(path: Path, texts: list[str], **extra_cols) -> None:
    data = {"text": texts}
    data.update(extra_cols)
    table = pa.table(data)
    pq.write_table(table, path)


@pytest.fixture
def tiny_tok(tmp_path: Path) -> Path:
    corpus = tmp_path / "tok_corpus.txt"
    lines = [
        "한국어 문서입니다. 파이프라인 테스트용 문장입니다.",
        "Another English sentence for the tokenizer training corpus.",
        "한글과 English를 함께 넣습니다. Digits 123.",
    ] * 40
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "tok.json"
    train_tokenizer(corpus, out, vocab_size=400, min_frequency=1)
    return out


def test_parquet_parsing_and_hangul(tmp_path: Path):
    path = tmp_path / "docs.parquet"
    texts = [
        "한글이 보존되어야 합니다. " * 5,
        "English only document for mixed corpus. " * 3,
    ]
    _write_parquet(path, texts, url=["http://a", "http://b"])
    docs = []
    for doc, err in iter_parquet_documents(path, source="fineweb2_ko", language="ko"):
        assert err is None
        assert doc is not None
        docs.append(doc)
    assert len(docs) == 2
    assert "한글" in docs[0].text
    assert docs[0].metadata.get("url") == "http://a"
    assert normalize_text(docs[0].text).startswith("한글")


def test_parquet_missing_text_column(tmp_path: Path):
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"body": ["x"]}), path)
    with pytest.raises(ValueError, match="text"):
        list(iter_parquet_documents(path, source="s"))


def test_audit_end_to_end(tmp_path: Path, tiny_tok: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Keep enough docs; include short + repeat + dup
    good = "한국어 일반 웹 문서입니다. 필터를 통과할 만큼 충분히 깁니다. " * 3
    texts = [good, good, "short", "ㅋ" * 80, "English article about science and history. " * 4]
    _write_parquet(raw / "000.parquet", texts)

    out = tmp_path / "audit.json"
    report = run_fineweb2_ko_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=out,
        min_chars=32,
        sample_count=2,
        progress_every=0,
    )
    assert out.is_file()
    assert report["input"]["raw_documents"] == 5
    assert report["cleaning"]["duplicates_removed"] >= 1
    assert report["cleaning"]["filtered_documents"] >= 1
    assert report["tokenizer"]["total_siat_tokens"] > 0
    assert report["tokenizer"]["eos_count"] == report["cleaning"]["kept_documents"]
    assert report["tokenizer"]["invalid_token_ids"] == 0
    assert report["scale_estimate"]["model_params"] == 36_837_888
    assert report["scale_estimate"]["estimated_tokens_needed"] == 36_837_888 * 20
    assert report["integrity"]["original_files_modified"] is False
    assert len(report["samples"]) <= 2
    # NFC / hangul
    assert report["cleaning"]["hangul_document_ratio"] >= 0.0


def test_invalid_token_detection(tmp_path: Path, tiny_tok: Path, monkeypatch: pytest.MonkeyPatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    text = "충분히 긴 한국어 테스트 문서입니다. " * 5
    _write_parquet(raw / "a.parquet", [text])

    class BadTok:
        def get_vocab_size(self):
            return 10

        def token_to_id(self, t):
            return 1

        def encode(self, text):
            class R:
                ids = [999]  # out of range

            return R()

    monkeypatch.setattr(
        "data.audit_fineweb2_ko.load_tokenizer", lambda path: BadTok()
    )
    monkeypatch.setattr(
        "data.audit_fineweb2_ko.sha256_file", lambda path: "deadbeef"
    )
    with pytest.raises(RuntimeError, match="Invalid token"):
        run_fineweb2_ko_audit(
            input_path=raw,
            tokenizer_path=tiny_tok,
            output_path=tmp_path / "out.json",
            progress_every=0,
        )
