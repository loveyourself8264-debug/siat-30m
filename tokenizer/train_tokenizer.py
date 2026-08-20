"""Train a Siat BPE tokenizer from raw .txt corpus files.

Design choices (kept intentionally simple):
- Algorithm: Hugging Face ``tokenizers`` BPE (not a hand-rolled merge loop).
- Normalization: Unicode NFC only — composes Hangul safely without lowercasing,
  stripping punctuation/digits, or forced jamo decomposition (NFKC avoided).
- Pre-tokenization: Metaspace (SentencePiece-style ``▁``) — preserves whitespace
  structure for Korean/English mixed text without a morphological analyzer.
- Unknown handling: BPE ``byte_fallback=True`` so rare Unicode tends to become
  byte tokens instead of collapsing to ``<|unk|>``.
- Training is corpus-driven; no pretrained tokenizer download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from tokenizer import DEFAULT_OUTPUT, DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS

UNK_TOKEN = "<|unk|>"


def iter_text_files(input_path: str | Path) -> Iterator[str]:
    """Yield non-empty lines from a .txt file or all .txt files under a directory.

    Files are opened one at a time and streamed line-by-line so large corpora
    need not be loaded entirely into memory.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer input path does not exist: {path.resolve()}"
        )

    if path.is_file():
        if path.suffix.lower() != ".txt":
            raise ValueError(
                f"Expected a .txt file, got {path.name!r} "
                f"(suffix={path.suffix!r})."
            )
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*.txt") if p.is_file())
        if not files:
            raise FileNotFoundError(
                f"No .txt files found under directory: {path.resolve()}"
            )
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {path}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield text


def build_tokenizer() -> Tokenizer:
    """Assemble an untrained BPE tokenizer with Siat defaults."""
    tokenizer = Tokenizer(
        models.BPE(unk_token=UNK_TOKEN, byte_fallback=True)
    )
    # NFC: compose characters (e.g. Hangul) without aggressive compatibility folds.
    tokenizer.normalizer = normalizers.NFC()
    # Metaspace: mark word boundaries with ▁; reversible via matching decoder.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always", split=True
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="▁", prepend_scheme="always", split=True
    )
    return tokenizer


def train_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a BPE tokenizer on .txt corpus, save JSON, and return the tokenizer.

    Parameters
    ----------
    input_path:
        A single ``.txt`` file or a directory containing ``.txt`` files.
    output_path:
        Destination JSON path (e.g. ``tokenizer/siat-tokenizer.json``).
    vocab_size:
        Target vocabulary size including special tokens. Default 32000 matches
        ``ModelConfig.siat_30m().vocab_size``; override for tiny fixtures.
    min_frequency:
        Minimum token frequency for BPE merges.
    """
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size must be > {len(SPECIAL_TOKENS)} "
            f"(number of special tokens), got {vocab_size}."
        )
    if min_frequency < 1:
        raise ValueError(f"min_frequency must be >= 1, got {min_frequency}.")

    # Probe that at least one non-empty line exists, then train from a fresh
    # iterator so ``train_from_iterator`` still streams the full corpus.
    line_iter = iter_text_files(input_path)
    try:
        first_line = next(line_iter)
    except StopIteration as exc:
        raise ValueError(
            f"No non-empty text lines found under: {Path(input_path).resolve()}"
        ) from exc

    def _training_lines() -> Iterator[str]:
        yield first_line
        yield from line_iter

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(_training_lines(), trainer=trainer)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    return tokenizer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Siat BPE tokenizer from .txt corpus files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .txt file or a directory of .txt files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Target vocabulary size (default: {DEFAULT_VOCAB_SIZE}).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for BPE merges (default: 2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tokenizer = train_tokenizer(
        input_path=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"Saved tokenizer to {Path(args.output).resolve()}")
    print(f"vocab_size (get_vocab_size) = {tokenizer.get_vocab_size()}")
    print("Special token IDs:")
    for token in SPECIAL_TOKENS:
        print(f"  {token}: {tokenizer.token_to_id(token)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
