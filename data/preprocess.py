"""Preprocess raw .txt corpora into tokenized train/val binary streams.

Pipeline:
  .txt documents (one file = one document)
    → deterministic document-level train/val split
    → Siat tokenizer encode + <|eos|> per document
    → NumPy .bin token streams + metadata.json

Does not load the full corpus into a Python list; documents are encoded and
flushed to disk in chunks.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from tokenizer import load_tokenizer

DEFAULT_TOKENIZER_PATH = Path("tokenizer/siat-tokenizer.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_VALIDATION_RATIO = 0.01
DEFAULT_SEED = 42
WRITE_BUFFER_TOKENS = 65_536
EOS_TOKEN = "<|eos|>"


def list_txt_files(input_path: str | Path) -> list[Path]:
    """Return sorted .txt file paths from a file or directory."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus input path does not exist: {path.resolve()}"
        )

    if path.is_file():
        if path.suffix.lower() != ".txt":
            raise ValueError(
                f"Expected a .txt file, got {path.name!r} "
                f"(suffix={path.suffix!r})."
            )
        return [path]

    if path.is_dir():
        files = sorted(p for p in path.rglob("*.txt") if p.is_file())
        if not files:
            raise FileNotFoundError(
                f"No .txt files found under directory: {path.resolve()}"
            )
        return files

    raise ValueError(f"Input path is neither a file nor a directory: {path}")


def choose_token_dtype(vocab_size: int) -> np.dtype:
    """Pick a compact integer dtype that can hold every token id.

    Siat's default vocab is 32_000, which fits in ``uint16`` (max 65_535).
    Larger vocabs use ``uint32`` so ids never wrap.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
    if vocab_size <= int(np.iinfo(np.uint16).max):
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def split_documents(
    paths: list[Path],
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], list[Path]]:
    """Document-level train/val split with a deterministic shuffle.

    Token-level shuffling is avoided to prevent leakage across the split.
    When there are at least two documents, both splits get at least one doc.
    """
    if not paths:
        raise ValueError("Cannot split an empty document list.")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio must be in [0.0, 1.0), got {validation_ratio}."
        )

    shuffled = list(paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 1 or validation_ratio == 0.0:
        return shuffled, []

    n_val = min(max(1, round(n * validation_ratio)), n - 1)
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]
    return train_paths, val_paths


def _read_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Document is empty after stripping: {path}")
    return text


def write_token_bin(
    paths: list[Path],
    tokenizer: Tokenizer,
    eos_id: int,
    out_bin: str | Path,
    dtype: np.dtype,
) -> int:
    """Encode documents sequentially, append EOS, and write a raw .bin file.

    Returns the total number of tokens written.
    """
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    buffer = np.empty(WRITE_BUFFER_TOKENS, dtype=dtype)
    buf_len = 0
    total = 0

    with out_bin.open("wb") as handle:
        def flush() -> None:
            nonlocal buf_len, total
            if buf_len == 0:
                return
            handle.write(buffer[:buf_len].tobytes())
            total += buf_len
            buf_len = 0

        for path in paths:
            ids = tokenizer.encode(_read_document(path)).ids
            ids = list(ids) + [eos_id]
            for token_id in ids:
                if token_id < 0 or token_id > int(np.iinfo(dtype).max):
                    raise ValueError(
                        f"Token id {token_id} does not fit dtype {dtype}."
                    )
                buffer[buf_len] = token_id
                buf_len += 1
                if buf_len == WRITE_BUFFER_TOKENS:
                    flush()
        flush()

    return total


def preprocess_corpus(
    input_path: str | Path,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run full preprocessing and write train.bin / val.bin / metadata.json."""
    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing required token {EOS_TOKEN!r}.")

    vocab_size = tokenizer.get_vocab_size()
    dtype = choose_token_dtype(vocab_size)

    documents = list_txt_files(input_path)
    train_paths, val_paths = split_documents(
        documents, validation_ratio=validation_ratio, seed=seed
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    metadata_path = output_dir / "metadata.json"

    train_tokens = write_token_bin(
        train_paths, tokenizer, eos_id, train_bin, dtype
    )
    val_tokens = 0
    if val_paths:
        val_tokens = write_token_bin(
            val_paths, tokenizer, eos_id, val_bin, dtype
        )
    elif val_bin.exists():
        val_bin.unlink()

    metadata: dict[str, Any] = {
        "tokenizer": str(Path(tokenizer_path).as_posix()),
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "eos_token_id": eos_id,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "num_train_documents": len(train_paths),
        "num_val_documents": len(val_paths),
        "train_bin": train_bin.name,
        "val_bin": val_bin.name if val_paths else None,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess .txt corpus into Siat train/val token binaries."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .txt file or a directory of .txt documents.",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_TOKENIZER_PATH),
        help=f"Siat tokenizer JSON (default: {DEFAULT_TOKENIZER_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help=f"Fraction of documents for validation (default: {DEFAULT_VALIDATION_RATIO}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Shuffle seed for document split (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = preprocess_corpus(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(f"Wrote processed data to {Path(args.output_dir).resolve()}")
    print(
        f"train_tokens={metadata['train_tokens']} "
        f"val_tokens={metadata['val_tokens']} "
        f"dtype={metadata['dtype']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
