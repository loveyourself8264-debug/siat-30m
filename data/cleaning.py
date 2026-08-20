"""Conservative text normalization for Siat pretraining documents.

Default Unicode form: NFC (preserves Korean syllable composition).
"""

from __future__ import annotations

import re
import unicodedata

# Keep newline and tab; drop other C0/C1 controls and NUL.
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize text conservatively for LLM pretraining.

    * Unicode NFC
    * Unify line endings to ``\\n``
    * Remove NUL / control chars except ``\\n`` and ``\\t``
    * Strip trailing whitespace per line
    * Collapse 3+ consecutive blank lines to 2
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)!r}.")

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def is_empty_after_normalize(text: str) -> bool:
    """True if normalized text is empty or whitespace-only."""
    return normalize_text(text) == ""
