"""Siat tokenizer package: load / inspect a trained BPE tokenizer.

Training lives in ``tokenizer.train_tokenizer`` (import the function from
that module). This package exposes the runtime load / special-token API for
encode/decode and later Dataset use.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

# Special tokens for Siat (shared by train + runtime).
# <|pad|> : padding in batched sequences (ignored by attention later)
# <|unk|> : true unknown / fallback when nothing else applies
# <|bos|> : beginning of sequence / document
# <|eos|> : end of sequence / document
SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
]

DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_OUTPUT = Path("tokenizer/siat-tokenizer.json")

__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_VOCAB_SIZE",
    "SPECIAL_TOKENS",
    "load_tokenizer",
    "special_token_ids",
]


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer previously saved with ``Tokenizer.save`` / train CLI."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path.resolve()}")
    return Tokenizer.from_file(str(path))


def special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Return ``{special_token: id}`` for all Siat special tokens.

    Raises ``ValueError`` if any required special token is missing.
    """
    ids: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Special token missing from tokenizer vocab: {token}")
        ids[token] = token_id
    return ids
