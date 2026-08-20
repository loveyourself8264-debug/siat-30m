"""Tests for conservative text normalization."""

from __future__ import annotations

import unicodedata

from data.cleaning import is_empty_after_normalize, normalize_text


def test_unicode_nfc():
    # Hangul jamo composition → NFC syllable if applicable; NFC is idempotent.
    text = "가나다"
    assert normalize_text(text) == unicodedata.normalize("NFC", text)


def test_line_endings_and_trailing_ws():
    raw = "hello  \r\nworld\t  \r\n"
    out = normalize_text(raw)
    assert "\r" not in out
    assert out == "hello\nworld"


def test_control_chars_removed_keep_tab_newline():
    raw = "a\x00b\nc\td\x07e"
    out = normalize_text(raw)
    assert "\x00" not in out
    assert "\x07" not in out
    assert "\n" in out
    assert "\t" in out


def test_blank_line_collapse():
    raw = "a\n\n\n\n\nb"
    out = normalize_text(raw)
    assert "\n\n\n" not in out
    assert out == "a\n\nb"


def test_empty_and_whitespace():
    assert is_empty_after_normalize("   \n\t  ")
    assert normalize_text("   ") == ""


def test_preserve_korean_english_digits_punct():
    raw = "한글 English 123 !?"
    out = normalize_text(raw)
    assert "한글" in out
    assert "English" in out
    assert "123" in out
    assert "!" in out
    # Must not lowercase
    assert "English" in out
