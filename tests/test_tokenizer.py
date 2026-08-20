"""Tests for Siat BPE tokenizer training, save/reload, and encode/decode."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer import (
    SPECIAL_TOKENS,
    load_tokenizer,
    special_token_ids,
)
from tokenizer.train_tokenizer import iter_text_files, train_tokenizer

FIXTURE_CORPUS = Path(__file__).resolve().parent / "fixtures" / "corpus"
TEST_VOCAB_SIZE = 256

SAMPLE_SENTENCES = [
    "안녕하세요. Siat는 작은 언어모델입니다.",
    "대한민국의 수도는 서울입니다.",
    "오늘 날씨는 좋습니다.",
    "Transformer is a neural network architecture.",
    "Siat는 30M parameter language model입니다.",
    "가격은 12,500원입니다.",
    "AI/LLM, Python, PyTorch를 공부하고 있습니다.",
]


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory: pytest.TempPathFactory):
    """Train once on fixture corpus with a small vocab for fast tests."""
    out = tmp_path_factory.mktemp("tok") / "siat-tokenizer.json"
    tok = train_tokenizer(
        input_path=FIXTURE_CORPUS,
        output_path=out,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )
    return tok, out


def test_iter_text_files_reads_directory():
    lines = list(iter_text_files(FIXTURE_CORPUS))
    assert len(lines) >= 5
    assert any("서울" in line for line in lines)
    assert any("Transformer" in line for line in lines)


def test_iter_text_files_missing_path():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list(iter_text_files(FIXTURE_CORPUS / "no_such_dir"))


def test_iter_text_files_rejects_non_txt(tmp_path: Path):
    other = tmp_path / "notes.md"
    other.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a .txt file"):
        list(iter_text_files(other))


def test_train_save_reload(trained_tokenizer):
    tok, path = trained_tokenizer
    assert path.is_file()
    assert tok.get_vocab_size() <= TEST_VOCAB_SIZE
    reloaded = load_tokenizer(path)
    assert reloaded.get_vocab_size() == tok.get_vocab_size()


def test_load_tokenizer_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        load_tokenizer(tmp_path / "missing.json")


def test_special_tokens_present_and_stable(trained_tokenizer):
    tok, path = trained_tokenizer
    before = special_token_ids(tok)
    assert set(before) == set(SPECIAL_TOKENS)
    for token, token_id in before.items():
        assert token_id is not None
        assert tok.token_to_id(token) == token_id

    reloaded = load_tokenizer(path)
    after = special_token_ids(reloaded)
    assert before == after


def test_encode_decode_roundtrip_ids_stable(trained_tokenizer):
    tok, path = trained_tokenizer
    reloaded = load_tokenizer(path)
    text = "Siat는 30M parameter language model입니다."
    ids_a = tok.encode(text).ids
    ids_b = reloaded.encode(text).ids
    assert ids_a == ids_b
    assert len(ids_a) > 0
    decoded = reloaded.decode(ids_b)
    # Metaspace may normalize leading/trailing whitespace; meaning must remain.
    assert "Siat" in decoded
    assert "30M" in decoded
    assert "언어모델" in decoded or "language" in decoded


@pytest.mark.parametrize("text", SAMPLE_SENTENCES)
def test_sample_sentences_encode_decode(trained_tokenizer, text: str):
    tok, _ = trained_tokenizer
    encoding = tok.encode(text)
    assert encoding.ids
    decoded = tok.decode(encoding.ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_korean_english_digits_punctuation(trained_tokenizer):
    tok, _ = trained_tokenizer
    korean = "오늘 날씨는 좋습니다."
    english = "Transformer is a neural network architecture."
    mixed = "가격은 12,500원입니다."

    for text in (korean, english, mixed):
        ids = tok.encode(text).ids
        decoded = tok.decode(ids)
        assert ids
        assert decoded

    mixed_decoded = tok.decode(tok.encode(mixed).ids)
    assert "12" in mixed_decoded or "12,500" in mixed_decoded
    assert "원" in mixed_decoded
    assert "," in mixed_decoded or "12,500" in mixed_decoded


def test_train_tokenizer_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        train_tokenizer(
            input_path=tmp_path / "absent",
            output_path=tmp_path / "out.json",
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
        )
