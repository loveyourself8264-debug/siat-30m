"""Automated FP32 pretraining smoke (tiny model + synthetic bins)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from train.smoke_test import (
    SmokeFailure,
    make_smoke_model_config,
    run_fp32_smoke,
    write_synthetic_bins,
)
from train.trainer import get_learning_rate


def _prepare_bins(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = make_smoke_model_config("tiny")
    # Enough tokens for seq=32, batch=2, drop_last, several steps.
    meta = write_synthetic_bins(
        tmp_path,
        vocab_size=cfg.vocab_size,
        train_tokens=2048,
        val_tokens=512,
        seed=0,
    )
    return (
        tmp_path / meta["train_bin"],
        tmp_path / meta["val_bin"],
        tmp_path / "metadata.json",
    )


def test_fp32_pretraining_smoke_e2e(tmp_path: Path):
    train_bin, val_bin, meta = _prepare_bins(tmp_path)
    ckpt_dir = tmp_path / "ckpts"
    jsonl = tmp_path / "smoke.jsonl"

    result = run_fp32_smoke(
        preset="tiny",
        train_data=train_bin,
        val_data=val_bin,
        metadata=meta,
        sequence_length=32,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=6,
        warmup_steps=2,
        checkpoint_step=3,
        checkpoint_dir=ckpt_dir,
        log_interval=1,
        val_interval=3,
        jsonl_path=jsonl,
        device="cpu",
        verbose=False,
    )

    assert result.status == "PASSED"
    assert result.optimizer_steps == 6
    assert result.tokens_processed > 0
    assert result.micro_steps == 6  # accum=1
    assert math_isfinite(result.initial_train_loss)
    assert math_isfinite(result.final_train_loss)
    assert result.final_val_loss is not None
    assert math_isfinite(result.final_val_loss)
    assert math_isfinite(result.last_grad_norm)
    assert math_isfinite(result.initial_lr)
    assert math_isfinite(result.final_lr)
    assert result.params_changed
    assert result.params_finite
    assert not result.nan_loss
    assert not result.inf_loss
    assert not result.nan_grad
    assert not result.inf_grad
    assert result.resume_successful
    assert result.resumed_optimizer_step == 3
    assert result.resumed_token_count > 0
    assert result.lr_continuity
    assert result.warmup_ok

    ckpt = Path(result.checkpoint_path)
    assert ckpt.is_file()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["optimizer_step"] == 3
    assert "model" in payload and "optimizer" in payload

    # LR rose during warmup (step 0 LR < step 1 LR for our schedule).
    from config import TrainConfig

    tc = TrainConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        weight_decay=0.1,
        warmup_steps=2,
        max_steps=6,
        max_grad_norm=1.0,
        seed=42,
    )
    assert get_learning_rate(0, tc) < get_learning_rate(1, tc)

    # JSONL has expected keys on train lines.
    assert jsonl.is_file()
    lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    train_rec = json.loads(lines[0])
    for key in (
        "step",
        "train_loss",
        "learning_rate",
        "grad_norm",
        "tokens_processed",
        "tokens_per_second",
    ):
        assert key in train_rec

    # At least one validation event in history or jsonl.
    val_hist = [h for h in result.history if "val_loss" in h]
    assert val_hist or any("val_loss" in json.loads(ln) for ln in lines)


def test_fp32_smoke_empty_data_raises(tmp_path: Path):
    cfg = make_smoke_model_config("tiny")
    # Too few tokens → empty dataset for seq_len=32
    write_synthetic_bins(
        tmp_path,
        vocab_size=cfg.vocab_size,
        train_tokens=10,
        val_tokens=10,
        seed=1,
    )
    try:
        run_fp32_smoke(
            preset="tiny",
            train_data=tmp_path / "train.bin",
            val_data=tmp_path / "val.bin",
            metadata=tmp_path / "metadata.json",
            sequence_length=32,
            batch_size=1,
            max_steps=4,
            warmup_steps=1,
            checkpoint_step=2,
            checkpoint_dir=tmp_path / "ckpts",
            device="cpu",
            verbose=False,
        )
        assert False, "expected SmokeFailure"
    except SmokeFailure as e:
        assert e.category == "DATA"


def math_isfinite(x: float) -> bool:
    return bool(torch.isfinite(torch.tensor(float(x))))
