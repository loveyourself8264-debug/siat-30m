"""Tests for BF16 autocast mixed precision (no GradScaler / no FP16)."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, TrainConfig
from model.model import SiatForCausalLM
from train.smoke_test import run_fp32_smoke, write_synthetic_bins
from train.trainer import (
    SiatTrainer,
    is_bf16_supported,
    require_bf16_support,
)


def _tiny_model() -> SiatForCausalLM:
    return SiatForCausalLM(
        ModelConfig(
            model_name="Siat",
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            max_seq_len=32,
            dropout=0.0,
            tie_embeddings=True,
        )
    )


def _train_cfg(**kwargs) -> TrainConfig:
    defaults = dict(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=2,
        max_steps=20,
        max_grad_norm=1.0,
        seed=42,
        precision="fp32",
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def _loader(ids: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    ds = TensorDataset(ids, labels)

    def collate(samples):
        return {
            "input_ids": torch.stack([s[0] for s in samples], dim=0),
            "labels": torch.stack([s[1] for s in samples], dim=0),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def test_precision_mode_validation():
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp16")
    with pytest.raises(ValueError, match="precision"):
        SiatTrainer(_tiny_model(), _train_cfg(), device="cpu", precision="auto")


def test_fp32_autocast_is_nullcontext():
    trainer = SiatTrainer(_tiny_model(), _train_cfg(precision="fp32"), device="cpu")
    assert trainer.precision == "fp32"
    ctx = trainer._autocast_context()
    assert isinstance(ctx, contextlib.nullcontext)


def test_bf16_unsupported_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "train.trainer.is_bf16_supported",
        lambda device: False,
    )
    with pytest.raises(RuntimeError, match="BF16"):
        require_bf16_support("cpu")
    with pytest.raises(RuntimeError, match="BF16"):
        SiatTrainer(_tiny_model(), _train_cfg(precision="bf16"), device="cpu")


def test_bf16_trainer_when_supported():
    if not is_bf16_supported("cpu") and not (
        torch.cuda.is_available() and is_bf16_supported("cuda")
    ):
        pytest.skip("BF16 not supported on this test device")
    device = "cuda" if torch.cuda.is_available() and is_bf16_supported("cuda") else "cpu"
    trainer = SiatTrainer(
        _tiny_model(), _train_cfg(precision="bf16"), device=device
    )
    assert trainer.precision == "bf16"
    assert not isinstance(trainer._autocast_context(), contextlib.nullcontext)

def test_fp32_backward_compat_train_and_validate():
    torch.manual_seed(0)
    model = _tiny_model()
    trainer = SiatTrainer(model, _train_cfg(precision="fp32"), device="cpu")
    ids = torch.randint(0, 64, (8, 8))
    labels = torch.randint(0, 64, (8, 8))
    loader = _loader(ids, labels, 2)
    hist = trainer.train(loader, log_interval=0, max_steps=2)
    assert len(hist) == 2
    assert torch.isfinite(torch.tensor(hist[-1]["train_loss"]))
    metrics = trainer.validate(loader)
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
    for p in model.parameters():
        assert p.dtype == torch.float32


def test_checkpoint_stores_precision(tmp_path: Path):
    model = _tiny_model()
    trainer = SiatTrainer(
        model, _train_cfg(precision="fp32"), device="cpu", model_config=None
    )
    path = trainer.save_checkpoint(tmp_path / "p.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["precision"] == "fp32"
    assert ckpt["train_config"]["precision"] == "fp32"


@pytest.mark.skipif(
    not is_bf16_supported("cpu")
    and not (torch.cuda.is_available() and is_bf16_supported("cuda")),
    reason="BF16 not supported on this test device",
)
def test_bf16_supported_smoke(tmp_path: Path):
    device = "cuda" if torch.cuda.is_available() and is_bf16_supported("cuda") else "cpu"
    cfg = ModelConfig(
        model_name="Siat",
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_dim=192,
        max_seq_len=64,
        dropout=0.0,
        tie_embeddings=True,
    )
    meta = write_synthetic_bins(
        tmp_path,
        vocab_size=cfg.vocab_size,
        train_tokens=2048,
        val_tokens=512,
        seed=0,
    )
    result = run_fp32_smoke(
        preset="tiny",
        train_data=tmp_path / meta["train_bin"],
        val_data=tmp_path / meta["val_bin"],
        metadata=tmp_path / "metadata.json",
        sequence_length=32,
        batch_size=2,
        max_steps=6,
        warmup_steps=2,
        checkpoint_step=3,
        checkpoint_dir=tmp_path / "ckpts",
        device=device,
        precision="bf16",
        verbose=False,
    )
    assert result.status == "PASSED"
    assert "BF16" in result.precision
    assert result.resume_successful
    assert not result.nan_loss
    assert result.params_finite
    assert Path(result.checkpoint_path).is_file()
    ckpt = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert ckpt["precision"] == "bf16"


@pytest.mark.skipif(
    not is_bf16_supported("cpu")
    and not (torch.cuda.is_available() and is_bf16_supported("cuda")),
    reason="BF16 not supported on this test device",
)
def test_fp32_vs_bf16_loss_same_order(tmp_path: Path):
    device = "cuda" if torch.cuda.is_available() and is_bf16_supported("cuda") else "cpu"
    write_synthetic_bins(
        tmp_path,
        vocab_size=128,
        train_tokens=2048,
        val_tokens=512,
        seed=1,
    )
    common = dict(
        preset="tiny",
        train_data=tmp_path / "train.bin",
        val_data=tmp_path / "val.bin",
        metadata=tmp_path / "metadata.json",
        sequence_length=32,
        batch_size=2,
        max_steps=4,
        warmup_steps=1,
        checkpoint_step=2,
        device=device,
        seed=7,
        verbose=False,
    )
    fp32 = run_fp32_smoke(
        **common,
        checkpoint_dir=tmp_path / "ck_fp32",
        precision="fp32",
    )
    bf16 = run_fp32_smoke(
        **common,
        checkpoint_dir=tmp_path / "ck_bf16",
        precision="bf16",
    )
    assert torch.isfinite(torch.tensor(fp32.final_train_loss))
    assert torch.isfinite(torch.tensor(bf16.final_train_loss))
    # Same order of magnitude (not bit-identical).
    ratio = max(fp32.final_train_loss, bf16.final_train_loss) / max(
        min(fp32.final_train_loss, bf16.final_train_loss), 1e-6
    )
    assert ratio < 10.0, (
        f"FP32={fp32.final_train_loss} BF16={bf16.final_train_loss} ratio={ratio}"
    )


@pytest.mark.skipif(
    not is_bf16_supported("cpu")
    and not (torch.cuda.is_available() and is_bf16_supported("cuda")),
    reason="BF16 not supported on this test device",
)
def test_bf16_validation_and_param_dtype():
    device = "cuda" if torch.cuda.is_available() and is_bf16_supported("cuda") else "cpu"
    model = _tiny_model()
    trainer = SiatTrainer(
        model, _train_cfg(precision="bf16"), device=device
    )
    ids = torch.randint(0, 64, (4, 8))
    labels = torch.randint(0, 64, (4, 8))
    metrics = trainer.validate(_loader(ids, labels, 2))
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
    for p in model.parameters():
        assert p.dtype == torch.float32
