"""End-to-end pretraining data pipeline tests (offline fixtures only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from config import ModelConfig, TrainConfig
from data.build_pretraining_data import build_pretraining_data
from data.dataset import SiatDataset, create_dataloader
from data.document import iter_jsonl_documents, load_manifest
from model.model import SiatForCausalLM
from tokenizer.train_tokenizer import train_tokenizer
from train.loss import causal_lm_loss
from train.trainer import SiatTrainer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "example_fixture.json"
FIXTURE_CORPUS = ROOT / "tests" / "fixtures" / "corpus"
PRETRAIN_RAW = ROOT / "tests" / "fixtures" / "pretrain_raw"


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a small offline tokenizer for pipeline tests."""
    out = tmp_path_factory.mktemp("tok") / "siat-test-tokenizer.json"
    # Combine corpus + pretrain raw text for enough lines
    train_dir = tmp_path_factory.mktemp("tok_corpus")
    texts: list[str] = []
    for p in sorted(FIXTURE_CORPUS.glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8"))
    for p in sorted(PRETRAIN_RAW.rglob("*.txt")):
        texts.append(p.read_text(encoding="utf-8"))
    blob = train_dir / "all.txt"
    blob.write_text("\n".join(texts) + "\n", encoding="utf-8")
    train_tokenizer(
        input_path=blob,
        output_path=out,
        vocab_size=500,
        min_frequency=1,
    )
    return out


def test_manifest_loads():
    data = load_manifest(MANIFEST)
    assert len(data["sources"]) >= 2


def test_jsonl_iterator():
    path = PRETRAIN_RAW / "english" / "en_extra.jsonl"
    docs = []
    errors = []
    for doc, err in iter_jsonl_documents(path, source="en_jsonl"):
        if err:
            errors.append(err)
        else:
            docs.append(doc)
    assert len(docs) == 2
    assert any("missing_field" in e for e in errors)


def test_dry_run(tmp_path: Path, tiny_tokenizer: Path):
    out = tmp_path / "dry"
    result = build_pretraining_data(
        manifest_path=MANIFEST,
        tokenizer_path=tiny_tokenizer,
        output_dir=out,
        validation_ratio=0.25,
        seed=42,
        vocab_size=500,
        min_chars=20,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert (out / "statistics.json").is_file()
    assert result["dedup"]["duplicates"] >= 1


def test_e2e_build_metadata_and_bins(tmp_path: Path, tiny_tokenizer: Path):
    out = tmp_path / "build"
    meta = build_pretraining_data(
        manifest_path=MANIFEST,
        tokenizer_path=tiny_tokenizer,
        output_dir=out,
        validation_ratio=0.25,
        seed=42,
        vocab_size=500,
        min_chars=20,
        max_oversample=2,
    )
    assert (out / "train.bin").is_file()
    assert (out / "val.bin").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "statistics.json").is_file()
    assert (out / "sources.json").is_file()

    assert meta["pipeline_version"] == "siat-data-v1"
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] == 500
    assert meta["tokenizer_sha256"]
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0
    assert meta["eos_count_train"] == meta["train_documents"]
    assert meta["eos_count_val"] == meta["val_documents"]
    assert meta["unicode_normalization"] == "NFC"

    stats = json.loads((out / "statistics.json").read_text(encoding="utf-8"))
    assert "too_short" in stats["filter_reasons"] or stats["filtered_documents"] >= 1
    assert stats["duplicates_removed"] >= 1

    # Token id range sanity via memmap
    tokens = np.memmap(
        out / "train.bin", dtype=meta["dtype"], mode="r", shape=(meta["train_tokens"],)
    )
    assert int(tokens.max()) < 500
    assert int(tokens.min()) >= 0


def test_reproducible_build(tmp_path: Path, tiny_tokenizer: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    kwargs = dict(
        manifest_path=MANIFEST,
        tokenizer_path=tiny_tokenizer,
        validation_ratio=0.25,
        seed=42,
        vocab_size=500,
        min_chars=20,
    )
    ma = build_pretraining_data(output_dir=a, **kwargs)
    mb = build_pretraining_data(output_dir=b, **kwargs)
    assert ma["fingerprint"] == mb["fingerprint"]
    assert ma["train_tokens"] == mb["train_tokens"]
    ta = np.fromfile(a / "train.bin", dtype=ma["dtype"])
    tb = np.fromfile(b / "train.bin", dtype=mb["dtype"])
    assert np.array_equal(ta, tb)


def test_siadataset_and_trainer_step(tmp_path: Path, tiny_tokenizer: Path):
    out = tmp_path / "compat"
    meta = build_pretraining_data(
        manifest_path=MANIFEST,
        tokenizer_path=tiny_tokenizer,
        output_dir=out,
        validation_ratio=0.25,
        seed=1,
        vocab_size=500,
        min_chars=20,
    )
    ds = SiatDataset(
        out / "train.bin",
        sequence_length=16,
        metadata_path=out / "metadata.json",
    )
    assert len(ds) >= 1
    loader = create_dataloader(ds, batch_size=1, shuffle=False, drop_last=False)
    batch = next(iter(loader))

    model = SiatForCausalLM(
        ModelConfig(
            model_name="Siat",
            vocab_size=500,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_dim=64,
            max_seq_len=64,
            dropout=0.0,
            tie_embeddings=True,
        )
    )
    trainer = SiatTrainer(
        model,
        TrainConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            weight_decay=0.0,
            warmup_steps=1,
            max_steps=2,
            max_grad_norm=1.0,
            seed=1,
            precision="fp32",
        ),
        device="cpu",
    )
    hist = trainer.train(loader, log_interval=0, max_steps=2)
    assert len(hist) == 2
    assert torch.isfinite(torch.tensor(hist[0]["train_loss"]))

    # Direct loss path
    logits = model(batch["input_ids"])
    loss = causal_lm_loss(logits, batch["labels"])
    assert torch.isfinite(loss)
