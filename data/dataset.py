"""PyTorch Dataset / DataLoader helpers for Siat causal LM training.

Reads preprocessed NumPy ``.bin`` token streams via ``numpy.memmap`` and yields
fixed-length ``input_ids`` / ``labels`` pairs with a one-token shift for
next-token prediction.

Indexing (non-overlapping chunks; incomplete tail is dropped)::

    sample i:
      input_ids = tokens[i*S : i*S + S]
      labels    = tokens[i*S + 1 : i*S + S + 1]

    len(dataset) = (n_tokens - 1) // sequence_length
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata file not found: {metadata_path.resolve()}"
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class SiatDataset(Dataset):
    """Memory-mapped token stream → fixed-length causal LM samples."""

    def __init__(
        self,
        bin_path: str | Path,
        sequence_length: int,
        *,
        metadata_path: str | Path | None = None,
        dtype: np.dtype | str | None = None,
        token_count: int | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError(
                f"sequence_length must be >= 1, got {sequence_length}."
            )

        self.bin_path = Path(bin_path)
        if not self.bin_path.is_file():
            raise FileNotFoundError(
                f"Token binary not found: {self.bin_path.resolve()}"
            )

        self.sequence_length = int(sequence_length)

        if metadata_path is not None:
            meta = _load_metadata(Path(metadata_path))
            dtype_name = meta["dtype"]
            # Prefer explicit counts from metadata for the matching split file.
            name = self.bin_path.name
            if name == meta.get("train_bin"):
                n_tokens = int(meta["train_tokens"])
            elif name == meta.get("val_bin"):
                n_tokens = int(meta["val_tokens"])
            elif token_count is not None:
                n_tokens = int(token_count)
            else:
                raise ValueError(
                    f"Cannot infer token count for {name!r} from metadata; "
                    "pass token_count= explicitly."
                )
            np_dtype = np.dtype(dtype_name)
        else:
            if dtype is None or token_count is None:
                raise ValueError(
                    "Provide metadata_path, or both dtype and token_count."
                )
            np_dtype = np.dtype(dtype)
            n_tokens = int(token_count)

        if n_tokens < 0:
            raise ValueError(f"token_count must be >= 0, got {n_tokens}.")

        expected_bytes = n_tokens * np_dtype.itemsize
        actual_bytes = self.bin_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Binary size mismatch for {self.bin_path}: "
                f"expected {expected_bytes} bytes "
                f"({n_tokens} × {np_dtype.itemsize}), got {actual_bytes}."
            )

        self.dtype = np_dtype
        self.n_tokens = n_tokens
        self.tokens = np.memmap(
            self.bin_path, dtype=self.dtype, mode="r", shape=(self.n_tokens,)
        )

    def __len__(self) -> int:
        # Each sample needs sequence_length + 1 raw tokens (input + next label).
        if self.n_tokens < self.sequence_length + 1:
            return 0
        return (self.n_tokens - 1) // self.sequence_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}."
            )

        start = index * self.sequence_length
        end = start + self.sequence_length
        # input: [start, end), labels: [start+1, end+1)
        input_ids = np.asarray(self.tokens[start:end], dtype=np.int64)
        labels = np.asarray(self.tokens[start + 1 : end + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(input_ids.copy()),
            "labels": torch.from_numpy(labels.copy()),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Build a simple PyTorch DataLoader for a ``SiatDataset``."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
