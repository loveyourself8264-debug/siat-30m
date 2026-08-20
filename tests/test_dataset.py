"""Tests for Siat preprocessing, Dataset, and DataLoader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset import SiatDataset, create_dataloader
from data.preprocess import (
    choose_token_dtype,
    list_txt_files,
    preprocess_corpus,
    split_documents,
    write_token_bin,
)
from tokenizer import load_tokenizer
from tokenizer.train_tokenizer import train_tokenizer

CORPUS_FOR_TOKENIZER = Path(__file__).resolve().parent / "fixtures" / "corpus"
SAMPLE_DOCS = Path(__file__).resolve().parent / "fixtures" / "sample_docs"
TEST_VOCAB_SIZE = 256


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tok") / "siat-tokenizer.json"
    train_tokenizer(
        input_path=CORPUS_FOR_TOKENIZER,
        output_path=out,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )
    return out


@pytest.fixture
def processed(tmp_path: Path, tokenizer_path: Path) -> dict:
    out_dir = tmp_path / "processed"
    metadata = preprocess_corpus(
        input_path=SAMPLE_DOCS,
        tokenizer_path=tokenizer_path,
        output_dir=out_dir,
        validation_ratio=0.25,
        seed=42,
    )
    metadata["_dir"] = out_dir
    return metadata


def test_list_txt_files_sorted():
    files = list_txt_files(SAMPLE_DOCS)
    assert len(files) >= 2
    assert files == sorted(files)
    assert all(p.suffix == ".txt" for p in files)


def test_list_txt_files_missing():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_txt_files(SAMPLE_DOCS / "missing_dir")


def test_choose_token_dtype():
    assert choose_token_dtype(32_000) == np.dtype(np.uint16)
    assert choose_token_dtype(65_535) == np.dtype(np.uint16)
    assert choose_token_dtype(65_536) == np.dtype(np.uint32)


def test_split_documents_deterministic():
    paths = list_txt_files(SAMPLE_DOCS)
    a_train, a_val = split_documents(paths, validation_ratio=0.25, seed=42)
    b_train, b_val = split_documents(paths, validation_ratio=0.25, seed=42)
    assert a_train == b_train
    assert a_val == b_val
    assert len(a_train) >= 1
    assert len(a_val) >= 1
    assert set(a_train).isdisjoint(a_val)
    assert len(a_train) + len(a_val) == len(paths)


def test_split_different_seed_differs():
    paths = list_txt_files(SAMPLE_DOCS)
    a_train, a_val = split_documents(paths, validation_ratio=0.25, seed=1)
    b_train, b_val = split_documents(paths, validation_ratio=0.25, seed=99)
    assert (a_train, a_val) != (b_train, b_val)


def test_preprocess_creates_bins_and_metadata(processed: dict, tokenizer_path: Path):
    out_dir = processed["_dir"]
    assert (out_dir / "train.bin").is_file()
    assert (out_dir / "val.bin").is_file()
    meta_path = out_dir / "metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] == load_tokenizer(tokenizer_path).get_vocab_size()
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0
    assert meta["eos_token_id"] is not None
    assert meta["num_train_documents"] >= 1
    assert meta["num_val_documents"] >= 1


def test_tokenizer_loading_success(tokenizer_path: Path):
    tok = load_tokenizer(tokenizer_path)
    assert tok.get_vocab_size() > 0
    assert tok.token_to_id("<|eos|>") is not None


def test_missing_tokenizer_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        preprocess_corpus(
            input_path=SAMPLE_DOCS,
            tokenizer_path=tmp_path / "no-tok.json",
            output_dir=tmp_path / "out",
        )


def test_eos_between_documents(tmp_path: Path, tokenizer_path: Path):
    tok = load_tokenizer(tokenizer_path)
    eos_id = tok.token_to_id("<|eos|>")
    assert eos_id is not None

    docs = [
        tmp_path / "a.txt",
        tmp_path / "b.txt",
    ]
    docs[0].write_text("첫 번째 문서", encoding="utf-8")
    docs[1].write_text("두 번째 문서", encoding="utf-8")

    out_bin = tmp_path / "docs.bin"
    dtype = choose_token_dtype(tok.get_vocab_size())
    n = write_token_bin(docs, tok, eos_id, out_bin, dtype)
    tokens = np.fromfile(out_bin, dtype=dtype)
    assert len(tokens) == n

    ids_a = tok.encode("첫 번째 문서").ids
    ids_b = tok.encode("두 번째 문서").ids
    expected = np.array(ids_a + [eos_id] + ids_b + [eos_id], dtype=dtype)
    assert np.array_equal(tokens, expected)
    # Documents are not concatenated without EOS.
    assert tokens[len(ids_a)] == eos_id
    assert tokens[-1] == eos_id


def test_siadataset_shapes_dtypes_and_shift(processed: dict):
    out_dir = processed["_dir"]
    seq_len = 8
    ds = SiatDataset(
        out_dir / "train.bin",
        sequence_length=seq_len,
        metadata_path=out_dir / "metadata.json",
    )
    assert len(ds) == (processed["train_tokens"] - 1) // seq_len
    assert len(ds) >= 1

    sample = ds[0]
    assert sample["input_ids"].shape == (seq_len,)
    assert sample["labels"].shape == (seq_len,)
    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long
    assert torch.equal(sample["input_ids"][1:], sample["labels"][:-1])


def test_len_off_by_one(tmp_path: Path):
    # 10 tokens, seq_len=4 → need 5 tokens per sample → len = (10-1)//4 = 2
    tokens = np.arange(10, dtype=np.uint16)
    bin_path = tmp_path / "t.bin"
    tokens.tofile(bin_path)
    ds = SiatDataset(bin_path, sequence_length=4, dtype=np.uint16, token_count=10)
    assert len(ds) == 2
    s0 = ds[0]
    assert torch.equal(s0["input_ids"], torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(s0["labels"], torch.tensor([1, 2, 3, 4], dtype=torch.long))
    s1 = ds[1]
    assert torch.equal(s1["input_ids"], torch.tensor([4, 5, 6, 7], dtype=torch.long))
    assert torch.equal(s1["labels"], torch.tensor([5, 6, 7, 8], dtype=torch.long))


def test_incomplete_sequence_dropped(tmp_path: Path):
    # 6 tokens, seq_len=4 → (6-1)//4 = 1; remainder tokens 4,5 unused (no pad)
    tokens = np.arange(6, dtype=np.uint16)
    bin_path = tmp_path / "short.bin"
    tokens.tofile(bin_path)
    ds = SiatDataset(bin_path, sequence_length=4, dtype="uint16", token_count=6)
    assert len(ds) == 1
    # Too short for even one sample
    tiny = tmp_path / "tiny.bin"
    np.arange(3, dtype=np.uint16).tofile(tiny)
    ds_tiny = SiatDataset(tiny, sequence_length=4, dtype="uint16", token_count=3)
    assert len(ds_tiny) == 0


def test_dataloader_batch_shape(processed: dict):
    out_dir = processed["_dir"]
    seq_len = 4
    ds = SiatDataset(
        out_dir / "train.bin",
        sequence_length=seq_len,
        metadata_path=out_dir / "metadata.json",
    )
    loader = create_dataloader(ds, batch_size=2, shuffle=True, drop_last=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] <= 2
    assert batch["input_ids"].shape[1] == seq_len
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_validation_dataloader_no_shuffle(processed: dict):
    out_dir = processed["_dir"]
    ds = SiatDataset(
        out_dir / "val.bin",
        sequence_length=4,
        metadata_path=out_dir / "metadata.json",
    )
    if len(ds) < 2:
        pytest.skip("val set too small for shuffle comparison")
    loader_a = create_dataloader(ds, batch_size=1, shuffle=False)
    loader_b = create_dataloader(ds, batch_size=1, shuffle=False)
    ids_a = [b["input_ids"].tolist() for b in loader_a]
    ids_b = [b["input_ids"].tolist() for b in loader_b]
    assert ids_a == ids_b


def test_missing_bin_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Token binary not found"):
        SiatDataset(
            tmp_path / "missing.bin",
            sequence_length=8,
            dtype="uint16",
            token_count=10,
        )


def test_invalid_sequence_length(processed: dict):
    out_dir = processed["_dir"]
    with pytest.raises(ValueError, match="sequence_length"):
        SiatDataset(
            out_dir / "train.bin",
            sequence_length=0,
            metadata_path=out_dir / "metadata.json",
        )


def test_same_seed_preprocess_stable(
    tmp_path: Path, tokenizer_path: Path
):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    meta_a = preprocess_corpus(
        SAMPLE_DOCS, tokenizer_path, out_a, validation_ratio=0.25, seed=7
    )
    meta_b = preprocess_corpus(
        SAMPLE_DOCS, tokenizer_path, out_b, validation_ratio=0.25, seed=7
    )
    assert meta_a["num_train_documents"] == meta_b["num_train_documents"]
    assert meta_a["num_val_documents"] == meta_b["num_val_documents"]
    assert meta_a["train_tokens"] == meta_b["train_tokens"]
    ta = np.fromfile(out_a / "train.bin", dtype=meta_a["dtype"])
    tb = np.fromfile(out_b / "train.bin", dtype=meta_b["dtype"])
    assert np.array_equal(ta, tb)
