"""Tests for generic Fast Corpus Audit (synthetic data only)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.audit_corpus import estimate_tokens_from_chars, run_corpus_audit
from data.document import iter_parquet_documents
from tokenizer.train_tokenizer import train_tokenizer


def _write_parquet(path: Path, texts: list[str], **extra_cols) -> None:
    data = {"text": texts}
    data.update(extra_cols)
    pq.write_table(pa.table(data), path)


@pytest.fixture
def tiny_tok(tmp_path: Path) -> Path:
    corpus = tmp_path / "tok_corpus.txt"
    lines = [
        "한국어 위키 문서입니다. 파이프라인 테스트용 문장입니다.",
        "Another English sentence for the tokenizer training corpus.",
        "한글과 English를 함께 넣습니다. Digits 123. math x^2 = y.",
    ] * 40
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "tok.json"
    train_tokenizer(corpus, out, vocab_size=400, min_frequency=1)
    return out


def _good_texts(n: int) -> list[str]:
    base = "한국어 백과사전 문서입니다. 충분히 긴 본문으로 필터를 통과합니다. "
    return [f"{base}문서번호 {i}. " * 3 for i in range(n)]


def test_title_provenance(tmp_path: Path):
    path = tmp_path / "docs.parquet"
    _write_parquet(
        path,
        ["한글 제목 본문입니다. " * 5],
        title=["테스트 문서"],
        id=["page-1"],
    )
    docs = []
    for doc, err in iter_parquet_documents(path, source="finewiki_ko", language="ko"):
        assert err is None
        docs.append(doc)
    assert docs[0].metadata.get("title") == "테스트 문서"
    assert docs[0].metadata.get("id") == "page-1"


def test_estimate_tokens_from_chars_eos():
    # 100 chars, 40 content tokens + 10 EOS => sample_tokens=50, sample_docs=10
    # cpt = 100/40 = 2.5; kept 1000 chars, 20 docs => 1000/2.5 + 20 = 420
    out = estimate_tokens_from_chars(
        kept_characters=1000,
        kept_documents=20,
        sample_characters=100,
        sample_tokens=50,
        sample_documents=10,
        per_doc_chars_per_token=[2.4, 2.5, 2.6],
    )
    assert out["is_estimate"] is True
    assert out["chars_per_token"] == pytest.approx(2.5)
    assert out["estimated_total_tokens"] == 420
    low, high = out["estimated_token_range"]
    assert low <= 420 <= high


def test_deterministic_sampling(tmp_path: Path, tiny_tok: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    texts = _good_texts(40)
    _write_parquet(raw / "a.parquet", texts)

    r1 = run_corpus_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=tmp_path / "o1.json",
        source="finewiki_ko",
        sample_documents=8,
        seed=42,
        progress_every=0,
    )
    r2 = run_corpus_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=tmp_path / "o2.json",
        source="finewiki_ko",
        sample_documents=8,
        seed=42,
        progress_every=0,
    )
    assert r1["sample_tokens"] == r2["sample_tokens"]
    assert r1["estimated_total_tokens"] == r2["estimated_total_tokens"]
    assert r1["token_sample"]["sample_characters"] == r2["token_sample"]["sample_characters"]


def test_eos_accounting(tmp_path: Path, tiny_tok: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    texts = _good_texts(12)
    _write_parquet(raw / "a.parquet", texts)
    report = run_corpus_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=tmp_path / "out.json",
        sample_documents=12,
        seed=7,
        progress_every=0,
    )
    n = report["sample_documents"]
    assert n == 12
    # Each sampled doc contributes exactly one EOS → tokens > docs
    assert report["sample_tokens"] >= n
    assert report["token_sample"]["eos_included"] is True


def test_fast_mode_does_not_full_tokenize(
    tmp_path: Path, tiny_tok: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    texts = _good_texts(30)
    _write_parquet(raw / "a.parquet", texts)

    encode_count = {"n": 0}
    real_load = __import__("data.audit_corpus", fromlist=["load_tokenizer"]).load_tokenizer

    class CountingTok:
        def __init__(self, inner):
            self._inner = inner

        def get_vocab_size(self):
            return self._inner.get_vocab_size()

        def token_to_id(self, t):
            return self._inner.token_to_id(t)

        def encode(self, text):
            encode_count["n"] += 1
            return self._inner.encode(text)

        def encode_batch(self, texts):
            encode_count["n"] += len(texts)
            return self._inner.encode_batch(texts)

    monkeypatch.setattr(
        "data.audit_corpus.load_tokenizer",
        lambda path: CountingTok(real_load(path)),
    )

    sample_n = 5
    report = run_corpus_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=tmp_path / "fast.json",
        sample_documents=sample_n,
        seed=42,
        full_token_audit=False,
        progress_every=0,
    )
    assert report["integrity"]["full_tokenization_performed"] is False
    assert encode_count["n"] == sample_n
    assert report["tokenizer"]["texts_encoded"] == sample_n
    assert report["mode"] == "fast"


def test_full_token_audit_opt_in(
    tmp_path: Path, tiny_tok: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    texts = _good_texts(15)
    _write_parquet(raw / "a.parquet", texts)

    encode_count = {"n": 0}
    from tokenizer import load_tokenizer as real_load

    class CountingTok:
        def __init__(self, inner):
            self._inner = inner

        def get_vocab_size(self):
            return self._inner.get_vocab_size()

        def token_to_id(self, t):
            return self._inner.token_to_id(t)

        def encode(self, text):
            encode_count["n"] += 1
            return self._inner.encode(text)

        def encode_batch(self, texts):
            encode_count["n"] += len(texts)
            return self._inner.encode_batch(texts)

    monkeypatch.setattr(
        "data.audit_corpus.load_tokenizer",
        lambda path: CountingTok(real_load(path)),
    )

    report = run_corpus_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=tmp_path / "full.json",
        sample_documents=3,
        full_token_audit=True,
        progress_every=0,
    )
    assert report["integrity"]["full_tokenization_performed"] is True
    assert report["mode"] == "full"
    assert encode_count["n"] == report["kept_documents"]
    assert report["estimate"]["is_estimate"] is False
    assert report["estimated_total_tokens"] > 0
    assert report["tokenizer"]["texts_encoded"] == report["kept_documents"]


def test_missing_text_column(tmp_path: Path, tiny_tok: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(pa.table({"body": ["x" * 40]}), raw / "bad.parquet")
    with pytest.raises(ValueError, match="text"):
        run_corpus_audit(
            input_path=raw,
            tokenizer_path=tiny_tok,
            output_path=tmp_path / "out.json",
            progress_every=0,
        )


def test_english_edu_metadata_and_filter(tmp_path: Path, tiny_tok: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    texts = [
        "This is a high quality educational English article about science and history. " * 3,
        "Another educational English document covering mathematics and literature carefully. " * 3,
        "Short English educational content that still passes the minimum length filter. ",
        "Yet another English wiki-style educational paragraph for the audit sample set. " * 2,
        "Final English document discussing biology, chemistry, and physics for learners. " * 2,
    ]
    n = len(texts)
    _write_parquet(
        raw / "edu.parquet",
        texts,
        language=["en"] * n,
        language_score=[0.99, 0.95, 0.90, 0.97, 0.93],
        int_score=[3, 4, 5, 4, 3],
        score=[2.7, 3.8, 4.5, 3.6, 2.9],
        token_count=[100, 120, 80, 110, 90],
    )
    report = run_corpus_audit(
        input_path=raw,
        tokenizer_path=tiny_tok,
        output_path=tmp_path / "edu.json",
        source="fineweb_edu_en",
        language="en",
        sample_documents=5,
        seed=42,
        progress_every=0,
    )
    assert report["kept_documents"] >= 4
    assert report["cleaning"]["filter_reasons"].get("low_alpha_hangul", 0) == 0
    assert report["language_statistics"] is not None
    assert report["language_statistics"]["english_ratio"] == 1.0
    edu = report["education_score_statistics"]
    assert edu is not None
    assert "3" in edu["by_int_score"]
    assert "4" in edu["by_int_score"]
    assert "5" in edu["by_int_score"]
    assert edu["score_ge_4_ratio"] > 0
    assert report["dataset_token_comparison"] is not None
    assert report["dataset_token_comparison"]["approx_ratio_siat_over_dataset"] > 0
    assert report["integrity"]["full_tokenization_performed"] is False
    assert report["estimate"]["is_estimate"] is True
