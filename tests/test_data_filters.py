"""Tests for heuristic quality filters."""

from __future__ import annotations

from data.document import Document
from data.filters import FilterConfig, FilterStats, filter_document


def _doc(text: str, source: str = "s", language: str | None = None) -> Document:
    return Document(text=text, source=source, language=language)


def test_too_short_and_empty():
    stats = FilterStats()
    assert filter_document(_doc("short"), FilterConfig(min_chars=32), stats)[1] == "too_short"
    assert filter_document(_doc("   \n"), FilterConfig(min_chars=1), stats)[1] == "empty"


def test_normal_korean_passes():
    text = "한국어 일반 문장입니다. 파이프라인 필터를 통과해야 합니다. " * 2
    kept, reason = filter_document(_doc(text), FilterConfig(min_chars=32))
    assert reason is None
    assert kept is not None
    assert "한국어" in kept.text


def test_normal_english_passes():
    text = "This English paragraph should pass the conservative quality filters easily."
    kept, reason = filter_document(
        _doc(text, language="en"), FilterConfig(min_chars=32)
    )
    assert reason is None
    assert kept is not None


def test_english_not_rejected_as_low_alpha_hangul():
    text = "This English paragraph should pass the conservative quality filters easily."
    kept, reason = filter_document(
        Document(text=text, source="s", language="en"),
        FilterConfig(min_chars=32),
    )
    assert kept is not None
    assert reason is None


def test_english_low_alpha_reason():
    # Mostly digits/symbols — low Latin letter ratio
    text = "1234567890 !!!!! ????? ###### ******** ........"
    stats = FilterStats()
    kept, reason = filter_document(
        Document(text=text, source="s", language="en"),
        FilterConfig(min_chars=10, enable_repetition=False, enable_punct=False),
        stats,
    )
    assert kept is None
    assert reason == "low_alpha"
    assert "low_alpha_hangul" not in stats.reasons


def test_korean_low_alpha_hangul_reason():
    text = "1234567890 !!!!! ????? ###### ******** ........"
    stats = FilterStats()
    kept, reason = filter_document(
        Document(text=text, source="s", language="ko"),
        FilterConfig(min_chars=10, enable_repetition=False, enable_punct=False),
        stats,
    )
    assert kept is None
    assert reason == "low_alpha_hangul"


def test_mixed_passes():
    text = "한글과 English가 섞인 문장입니다. Digits 42 and symbols OK."
    kept, reason = filter_document(_doc(text), FilterConfig(min_chars=32))
    assert reason is None


def test_excessive_repetition():
    text = "ㅋ" * 80
    kept, reason = filter_document(
        _doc(text), FilterConfig(min_chars=10, max_repeat_ratio=0.4)
    )
    assert kept is None
    assert reason == "repetition"


def test_excessive_punctuation():
    text = "!!!!!!!!????????????##########!!!!!!!!!!!"
    kept, reason = filter_document(
        _doc(text),
        FilterConfig(
            min_chars=10,
            enable_repetition=False,
            min_alpha_hangul_ratio=0.0,
            max_punct_ratio=0.3,
        ),
    )
    assert kept is None
    assert reason == "excessive_punctuation"
