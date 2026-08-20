"""Siat data loading and preprocessing package.

Public entrypoints:
* ``SiatDataset`` / ``create_dataloader``
* ``python -m data.preprocess`` (simple .txt path)
* ``python -m data.build_pretraining_data`` (manifest multi-source pipeline)
"""

from __future__ import annotations

from data.dataset import SiatDataset, create_dataloader

__all__ = [
    "SiatDataset",
    "create_dataloader",
]
