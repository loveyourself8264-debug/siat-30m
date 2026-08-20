"""Explainable heuristic quality filters for pretraining documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from data.cleaning import normalize_text
from data.document import Document

# Hangul syllables + jamo + latin letters (Korean / mixed corpora)
_ALPHA_HANGUL_RE = re.compile(r"[A-Za-z\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]")
# Latin letters only (English corpora)
_LATIN_RE = re.compile(r"[A-Za-z]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s")


def _is_english_language(language: str | None) -> bool:
    if not language:
        return False
    lang = language.strip().lower()
    return lang == "en" or lang.startswith("en-") or lang == "english"


@dataclass
class FilterConfig:
    """Thresholds for quality filters (conservative defaults)."""

    min_chars: int = 32
    max_chars: int | None = None
    max_repeat_ratio: float = 0.40
    min_alpha_hangul_ratio: float = 0.30
    max_whitespace_ratio: float = 0.50
    max_punct_ratio: float = 0.45
    enable_repetition: bool = True
    enable_alpha_hangul: bool = True
    enable_whitespace: bool = True
    enable_punct: bool = True


@dataclass
class FilterStats:
    """Per-reason rejection counts."""

    reasons: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)
    max_samples_per_reason: int = 3

    def record(self, reason: str, preview: str = "") -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        if preview and reason not in self.samples:
            self.samples[reason] = []
        if preview and len(self.samples.get(reason, [])) < self.max_samples_per_reason:
            self.samples.setdefault(reason, []).append(preview[:120])


def _max_run_ratio(text: str) -> float:
    if not text:
        return 1.0
    best = 1
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best / len(text)


def filter_document(
    doc: Document,
    config: FilterConfig | None = None,
    stats: FilterStats | None = None,
) -> tuple[Document | None, str | None]:
    """Return ``(cleaned_doc, None)`` or ``(None, reason)``.

    Applies ``normalize_text`` then heuristic checks. Does not mutate ``doc``.

    Alphabetic ratio check is language-aware:
    * ``en`` → Latin letters only; reject reason ``low_alpha``
    * otherwise → Hangul+Latin; reject reason ``low_alpha_hangul``
    """
    cfg = config or FilterConfig()
    text = normalize_text(doc.text)
    preview = text[:80].replace("\n", " ")

    def reject(reason: str) -> tuple[None, str]:
        if stats is not None:
            stats.record(reason, preview)
        return None, reason

    if not text:
        return reject("empty")

    if len(text) < cfg.min_chars:
        return reject("too_short")

    if cfg.max_chars is not None and len(text) > cfg.max_chars:
        return reject("too_long")

    n = len(text)
    if cfg.enable_repetition and _max_run_ratio(text) > cfg.max_repeat_ratio:
        return reject("repetition")

    if cfg.enable_whitespace:
        ws = len(_WS_RE.findall(text))
        if ws / n > cfg.max_whitespace_ratio:
            return reject("excessive_whitespace")

    if cfg.enable_punct:
        punct = len(_PUNCT_RE.findall(text))
        if punct / n > cfg.max_punct_ratio:
            return reject("excessive_punctuation")

    if cfg.enable_alpha_hangul:
        if _is_english_language(doc.language):
            alpha = len(_LATIN_RE.findall(text))
            if alpha / n < cfg.min_alpha_hangul_ratio:
                return reject("low_alpha")
        else:
            alpha = len(_ALPHA_HANGUL_RE.findall(text))
            if alpha / n < cfg.min_alpha_hangul_ratio:
                return reject("low_alpha_hangul")

    cleaned = Document(
        text=text,
        source=doc.source,
        language=doc.language,
        document_id=doc.document_id,
        metadata=dict(doc.metadata),
    )
    return cleaned, None


def filter_config_from_dict(data: dict[str, Any] | None) -> FilterConfig:
    if not data:
        return FilterConfig()
    allowed = {f.name for f in fields(FilterConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return FilterConfig(**kwargs)
