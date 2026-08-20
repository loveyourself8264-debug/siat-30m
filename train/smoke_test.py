"""Pretraining / precision smoke test for Siat.

Verifies Dataset → Trainer → val → checkpoint → resume end-to-end.
Supports ``--precision fp32`` (default) and ``--precision bf16`` (autocast).
Does not implement FP16, GradScaler, distributed training, or full pretraining.

Usage::

    python -m train.smoke_test --model tiny --synthetic-dir data/smoke/processed

    python -m train.smoke_test \\
        --precision bf16 \\
        --model siat_30m \\
        --train-data data/processed/train.bin \\
        --val-data data/processed/val.bin \\
        --sequence-length 256 \\
        --batch-size 2 \\
        --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config import ModelConfig, TrainConfig
from data.dataset import SiatDataset, create_dataloader
from model.model import SiatForCausalLM
from train.trainer import SiatTrainer, get_learning_rate, is_bf16_supported

SMOKE_SEED = 42


class SmokeFailure(RuntimeError):
    """Smoke test failure with a classified category."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"[{category}] {message}")


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def peak_param_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    return torch.float32


def assert_params_finite(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise SmokeFailure(
                "NUMERICAL_STABILITY",
                f"Non-finite parameter: {name}",
            )


def write_synthetic_bins(
    out_dir: str | Path,
    *,
    vocab_size: int,
    train_tokens: int,
    val_tokens: int,
    seed: int = SMOKE_SEED,
) -> dict[str, Any]:
    """Write uint16 train.bin / val.bin / metadata.json for smoke tests."""
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}.")
    if train_tokens < 2 or val_tokens < 2:
        raise ValueError("train_tokens and val_tokens must be >= 2.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dtype = np.dtype(np.uint16)
    if vocab_size > int(np.iinfo(dtype).max):
        dtype = np.dtype(np.uint32)

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    meta_path = out_dir / "metadata.json"

    train_arr = rng.integers(0, vocab_size, size=train_tokens, dtype=dtype)
    val_arr = rng.integers(0, vocab_size, size=val_tokens, dtype=dtype)
    train_arr.tofile(train_bin)
    val_arr.tofile(val_bin)

    metadata: dict[str, Any] = {
        "tokenizer": "synthetic",
        "vocab_size": vocab_size,
        "dtype": dtype.name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "eos_token_id": None,
        "validation_ratio": val_tokens / max(train_tokens + val_tokens, 1),
        "seed": seed,
        "num_train_documents": 1,
        "num_val_documents": 1,
        "train_bin": train_bin.name,
        "val_bin": val_bin.name,
        "synthetic": True,
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def make_smoke_model_config(preset: str) -> ModelConfig:
    """Return model config for smoke. ``siat_30m`` architecture is unchanged."""
    if preset == "tiny":
        return ModelConfig(
            model_name="Siat",
            vocab_size=128,
            d_model=64,
            n_layers=2,
            n_heads=4,
            ffn_dim=192,
            max_seq_len=64,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
        )
    if preset == "siat_30m":
        return ModelConfig.siat_30m()
    raise ValueError(f"Unknown model preset: {preset!r}")


def make_smoke_train_config(
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = SMOKE_SEED,
    precision: str = "fp32",
) -> TrainConfig:
    """Smoke-only TrainConfig override (does not mutate production presets)."""
    return TrainConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        precision=precision,
    )


def _build_dataset(
    bin_path: Path,
    sequence_length: int,
    metadata: Path | None,
) -> SiatDataset:
    meta = metadata if metadata is not None and metadata.is_file() else None
    if meta is None:
        sibling = bin_path.parent / "metadata.json"
        meta = sibling if sibling.is_file() else None
    return SiatDataset(
        bin_path,
        sequence_length=sequence_length,
        metadata_path=meta,
    )


def _window_mean(losses: list[float], window: int) -> float:
    if not losses:
        return float("nan")
    w = min(window, len(losses))
    return float(sum(losses[:w]) / w) if window > 0 else float("nan")


def _last_window_mean(losses: list[float], window: int) -> float:
    if not losses:
        return float("nan")
    w = min(window, len(losses))
    return float(sum(losses[-w:]) / w)


@dataclass
class SmokeResult:
    status: str
    failure_category: str | None
    device: str
    precision: str
    model_dtype: str
    preset: str
    parameters: int
    sequence_length: int
    batch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    optimizer_steps: int
    micro_steps: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    train_tokens: int
    val_tokens: int
    train_samples: int
    val_samples: int
    initial_train_loss: float
    final_train_loss: float
    first_window_loss: float
    last_window_loss: float
    final_val_loss: float | None
    final_perplexity: float | None
    last_grad_norm: float
    tokens_processed: int
    average_tokens_per_sec: float
    initial_lr: float
    peak_lr: float
    final_lr: float
    warmup_ok: bool
    cosine_ok: bool
    checkpoint_path: str
    checkpoint_step: int
    resume_successful: bool
    resumed_optimizer_step: int
    resumed_token_count: int
    lr_continuity: bool
    params_changed: bool
    params_finite: bool
    nan_loss: bool
    inf_loss: bool
    nan_grad: bool
    inf_grad: bool
    elapsed_sec: float
    history: list[dict[str, float]]


def run_fp32_smoke(
    *,
    preset: str = "tiny",
    train_data: str | Path,
    val_data: str | Path,
    metadata: str | Path | None = None,
    sequence_length: int | None = None,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    max_steps: int = 8,
    warmup_steps: int = 2,
    checkpoint_step: int | None = None,
    checkpoint_dir: str | Path,
    log_interval: int = 1,
    val_interval: int | None = None,
    jsonl_path: str | Path | None = None,
    device: str | None = None,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = SMOKE_SEED,
    verbose: bool = True,
    precision: str = "fp32",
) -> SmokeResult:
    """Run smoke: train → validate → checkpoint → resume → continue.

    ``precision`` is ``fp32`` (default) or ``bf16`` (autocast; GradScaler unused).
    """
    t0 = time.perf_counter()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    if precision not in ("fp32", "bf16"):
        raise SmokeFailure(
            "NUMERICAL_STABILITY",
            f"precision must be 'fp32' or 'bf16', got {precision!r}.",
        )
    if precision == "bf16" and not is_bf16_supported(device_t):
        raise SmokeFailure(
            "BF16_UNSUPPORTED",
            "BF16 precision was requested but is not supported on the "
            f"selected device ({device_t}).",
        )

    model_cfg = make_smoke_model_config(preset)
    seq_len = (
        model_cfg.max_seq_len if sequence_length is None else int(sequence_length)
    )
    if seq_len < 1 or seq_len > model_cfg.max_seq_len:
        raise SmokeFailure(
            "DATA",
            f"sequence_length={seq_len} must be in [1, {model_cfg.max_seq_len}].",
        )

    if checkpoint_step is None:
        checkpoint_step = max(1, max_steps // 2)
    if checkpoint_step < 1 or checkpoint_step >= max_steps:
        raise SmokeFailure(
            "CHECKPOINT",
            f"checkpoint_step ({checkpoint_step}) must satisfy "
            f"1 <= checkpoint_step < max_steps ({max_steps}).",
        )
    if val_interval is None:
        val_interval = checkpoint_step

    train_cfg = make_smoke_train_config(
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        seed=seed,
        precision=precision,
    )

    torch.manual_seed(seed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device_t)

    train_bin = Path(train_data)
    val_bin = Path(val_data)
    meta_path = Path(metadata) if metadata else None

    try:
        train_ds = _build_dataset(train_bin, seq_len, meta_path)
        val_ds = _build_dataset(val_bin, seq_len, meta_path)
    except Exception as e:
        raise SmokeFailure("DATA", str(e)) from e

    if len(train_ds) == 0:
        raise SmokeFailure(
            "DATA",
            f"Train dataset empty for sequence_length={seq_len}: {train_bin}",
        )
    if len(val_ds) == 0:
        raise SmokeFailure(
            "DATA",
            f"Val dataset empty for sequence_length={seq_len}: {val_bin}",
        )

    train_loader = create_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{checkpoint_step:06d}.pt"

    try:
        model = SiatForCausalLM(model_cfg)
        n_params = count_params(model)
        dtype = peak_param_dtype(model)
        if dtype != torch.float32:
            raise SmokeFailure(
                "NUMERICAL_STABILITY",
                f"Expected FP32 parameters, got {dtype}.",
            )

        trainer = SiatTrainer(
            model, train_cfg, device=device_t, model_config=model_cfg
        )
    except torch.cuda.OutOfMemoryError as e:
        raise SmokeFailure(
            "DEVICE_MEMORY",
            f"OOM during model init. batch_size={batch_size} "
            f"sequence_length={seq_len} device={device}",
        ) from e
    except RuntimeError as e:
        if "BF16" in str(e):
            raise SmokeFailure("BF16_UNSUPPORTED", str(e)) from e
        raise

    precision_label = (
        "BF16 autocast" if precision == "bf16" else "FP32"
    )
    if verbose:
        print("Device:", device)
        print(f"Precision: {precision_label}")
        print(f"Parameter dtype: {dtype}")
        print(f"Model: {model_cfg.model_name}")
        print(
            f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)"
        )
        print(f"Preset: {preset}")
        print(f"sequence length: {seq_len}")
        print(f"batch size: {batch_size}")
        print(f"gradient accumulation: {gradient_accumulation_steps}")
        print(
            f"effective batch size: "
            f"{batch_size * gradient_accumulation_steps}"
        )
        print(f"optimizer steps: {max_steps}")
        print(f"learning rate: {learning_rate}")
        print(f"warmup steps: {warmup_steps}")
        print(f"weight decay: {weight_decay}")
        print(f"max grad norm: {max_grad_norm}")
        print(f"train dataset tokens: {train_ds.n_tokens}")
        print(f"validation dataset tokens: {val_ds.n_tokens}")
        print(f"train samples: {len(train_ds)}")
        print(f"validation samples: {len(val_ds)}")
        print("-" * 40)

    # Snapshot one weight for change detection.
    with torch.no_grad():
        snap_name, snap_tensor = next(model.named_parameters())
        snap_before = snap_tensor.detach().cpu().clone()

    history: list[dict[str, float]] = []
    nan_loss = inf_loss = nan_grad = inf_grad = False
    final_val_loss: float | None = None
    final_ppl: float | None = None
    resume_successful = False
    resumed_step = 0
    resumed_tokens = 0
    lr_continuity = False

    try:
        if verbose:
            print(f"Phase 1: train to step {checkpoint_step}")
        hist1 = trainer.train(
            train_loader,
            log_interval=log_interval,
            max_steps=checkpoint_step,
            val_dataloader=val_loader,
            val_interval=val_interval,
            val_max_batches=4,
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=checkpoint_step,
            jsonl_path=jsonl_path,
        )
        history.extend(hist1)

        if not ckpt_path.is_file():
            # Trainer also writes latest.pt; ensure step file exists.
            trainer.save_checkpoint(ckpt_path)
            trainer.save_checkpoint(ckpt_dir / "latest.pt")
        if verbose:
            print(f"checkpoint saved: {ckpt_path}")

        phase1_step = trainer.optimizer_step
        phase1_tokens = trainer.tokens_processed
        expected_next_lr = get_learning_rate(phase1_step, train_cfg)

        if verbose:
            print(f"resumed from step {phase1_step}")
            print("Phase 2: new trainer + load + continue")

        model2 = SiatForCausalLM(model_cfg)
        trainer2 = SiatTrainer(
            model2, replace(train_cfg), device=device_t, model_config=model_cfg
        )
        trainer2.load_checkpoint(ckpt_path)
        resume_successful = True
        resumed_step = trainer2.optimizer_step
        resumed_tokens = trainer2.tokens_processed

        if resumed_step != phase1_step:
            raise SmokeFailure(
                "RESUME",
                f"optimizer_step mismatch: got {resumed_step}, "
                f"expected {phase1_step}",
            )
        if resumed_tokens != phase1_tokens:
            raise SmokeFailure(
                "RESUME",
                f"tokens_processed mismatch: got {resumed_tokens}, "
                f"expected {phase1_tokens}",
            )

        lr_after = get_learning_rate(trainer2.optimizer_step, train_cfg)
        lr_continuity = abs(lr_after - expected_next_lr) < 1e-12
        if not lr_continuity:
            raise SmokeFailure(
                "LR_SCHEDULER",
                f"LR discontinuity: resume LR {lr_after} vs expected "
                f"{expected_next_lr}",
            )
        if verbose:
            print(
                f"Resume OK | step={resumed_step} tokens={resumed_tokens} "
                f"next_lr={lr_after:.2e}"
            )

        hist2 = trainer2.train(
            train_loader,
            log_interval=log_interval,
            max_steps=max_steps,
            val_dataloader=val_loader,
            val_interval=val_interval,
            val_max_batches=4,
            checkpoint_dir=None,
            checkpoint_interval=0,
            jsonl_path=jsonl_path,
        )
        history.extend(hist2)
        model = model2
        trainer = trainer2

    except SmokeFailure:
        raise
    except torch.cuda.OutOfMemoryError as e:
        raise SmokeFailure(
            "DEVICE_MEMORY",
            f"OOM during training. batch_size={batch_size} "
            f"sequence_length={seq_len} device={device}",
        ) from e
    except Exception as e:
        msg = str(e).lower()
        if "val" in msg:
            cat = "VALIDATION"
        elif "checkpoint" in msg or "file" in msg:
            cat = "CHECKPOINT"
        elif "loss" in msg:
            cat = "LOSS"
        else:
            cat = "OPTIMIZER"
        raise SmokeFailure(cat, str(e)) from e

    train_losses = [float(h["train_loss"]) for h in history if "train_loss" in h]
    grad_norms = [float(h["grad_norm"]) for h in history if "grad_norm" in h]
    lrs = [float(h["learning_rate"]) for h in history if "learning_rate" in h]

    for loss in train_losses:
        if math.isnan(loss):
            nan_loss = True
        if math.isinf(loss):
            inf_loss = True
    for g in grad_norms:
        if math.isnan(g):
            nan_grad = True
        if math.isinf(g):
            inf_grad = True

    if nan_loss or inf_loss or nan_grad or inf_grad:
        raise SmokeFailure(
            "NUMERICAL_STABILITY",
            f"Non-finite metrics: nan_loss={nan_loss} inf_loss={inf_loss} "
            f"nan_grad={nan_grad} inf_grad={inf_grad}",
        )

    # Validation at least once (from history or final call).
    val_events = [h for h in history if "val_loss" in h]
    if not val_events:
        val_metrics = trainer.validate(val_loader, max_batches=4)
        final_val_loss = float(val_metrics["val_loss"])
        final_ppl = float(val_metrics["perplexity"])
    else:
        final_val_loss = float(val_events[-1]["val_loss"])
        final_ppl = float(val_events[-1]["perplexity"])

    if not math.isfinite(final_val_loss):
        raise SmokeFailure(
            "VALIDATION", f"Non-finite val_loss: {final_val_loss}"
        )

    assert_params_finite(model)
    params_finite = True

    with torch.no_grad():
        after = dict(model.named_parameters())[snap_name].detach().cpu()
        params_changed = not torch.equal(snap_before, after)
    if not params_changed:
        raise SmokeFailure(
            "OPTIMIZER",
            f"Parameter {snap_name} unchanged after training.",
        )

    if trainer.optimizer_step != max_steps:
        raise SmokeFailure(
            "OPTIMIZER",
            f"Expected optimizer_step={max_steps}, got {trainer.optimizer_step}",
        )

    # LR schedule smoke: warmup rise + later not stuck below first if max>warmup
    initial_lr = lrs[0] if lrs else float("nan")
    peak_lr_seen = max(lrs) if lrs else float("nan")
    final_lr = lrs[-1] if lrs else float("nan")
    warmup_ok = True
    cosine_ok = True
    if warmup_steps >= 2 and len(lrs) >= 2:
        # First few steps should be non-decreasing during early warmup.
        warmup_slice = lrs[: min(warmup_steps, len(lrs))]
        warmup_ok = all(
            warmup_slice[i] <= warmup_slice[i + 1] + 1e-15
            for i in range(len(warmup_slice) - 1)
        )
    if max_steps > warmup_steps + 1 and len(lrs) > warmup_steps + 1:
        # After warmup, LR should eventually be <= peak.
        cosine_ok = final_lr <= peak_lr_seen + 1e-12

    if not warmup_ok:
        raise SmokeFailure("LR_SCHEDULER", "Warmup LR did not rise as expected.")

    window = max(1, min(3, len(train_losses)))
    first_w = _window_mean(train_losses, window)
    last_w = _last_window_mean(train_losses, window)
    initial_train_loss = train_losses[0] if train_losses else float("nan")
    final_train_loss = train_losses[-1] if train_losses else float("nan")
    last_grad = grad_norms[-1] if grad_norms else float("nan")

    elapsed = time.perf_counter() - t0
    avg_tok_s = (
        trainer.tokens_processed / elapsed if elapsed > 0 else 0.0
    )

    if verbose and device_t.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device_t) / (1024**2)
        print(f"Peak CUDA allocated: {peak_mem:.1f} MiB")

    result = SmokeResult(
        status="PASSED",
        failure_category=None,
        device=str(device),
        precision=precision_label,
        model_dtype=str(dtype),
        preset=preset,
        parameters=n_params,
        sequence_length=seq_len,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation_steps,
        effective_batch_size=batch_size * gradient_accumulation_steps,
        optimizer_steps=trainer.optimizer_step,
        micro_steps=trainer.micro_step,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        train_tokens=train_ds.n_tokens,
        val_tokens=val_ds.n_tokens,
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        first_window_loss=first_w,
        last_window_loss=last_w,
        final_val_loss=final_val_loss,
        final_perplexity=final_ppl,
        last_grad_norm=last_grad,
        tokens_processed=trainer.tokens_processed,
        average_tokens_per_sec=avg_tok_s,
        initial_lr=initial_lr,
        peak_lr=peak_lr_seen,
        final_lr=final_lr,
        warmup_ok=warmup_ok,
        cosine_ok=cosine_ok,
        checkpoint_path=str(ckpt_path),
        checkpoint_step=checkpoint_step,
        resume_successful=resume_successful,
        resumed_optimizer_step=resumed_step,
        resumed_token_count=resumed_tokens,
        lr_continuity=lr_continuity,
        params_changed=params_changed,
        params_finite=params_finite,
        nan_loss=nan_loss,
        inf_loss=inf_loss,
        nan_grad=nan_grad,
        inf_grad=inf_grad,
        elapsed_sec=elapsed,
        history=history,
    )

    if verbose:
        print_smoke_summary(result)
    return result


def print_smoke_summary(result: SmokeResult) -> None:
    ppl = result.final_perplexity
    ppl_s = f"{ppl:.4f}" if ppl is not None and math.isfinite(ppl) else str(ppl)
    val_s = (
        f"{result.final_val_loss:.4f}"
        if result.final_val_loss is not None
        else "n/a"
    )
    print()
    print("Siat Pretraining Smoke Test")
    print("--------------------------------")
    print(f"Status: {result.status}")
    print(f"Precision: {result.precision}")
    print(f"Optimizer steps: {result.optimizer_steps}")
    print(f"Tokens processed: {result.tokens_processed}")
    print(f"Initial train loss: {result.initial_train_loss:.4f}")
    print(f"Final train loss: {result.final_train_loss:.4f}")
    print(f"Final validation loss: {val_s}")
    print(f"Final perplexity: {ppl_s}")
    print(f"Peak/Final learning rate: {result.peak_lr:.2e} / {result.final_lr:.2e}")
    print(f"Last grad norm: {result.last_grad_norm:.4f}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Resume tested: {result.resume_successful}")
    print(f"Parameters finite: {result.params_finite}")
    print(f"Elapsed: {result.elapsed_sec:.2f}s")
    print(f"Average tokens/sec: {result.average_tokens_per_sec:.0f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Siat pretraining smoke (fp32 / bf16 autocast; no FP16/DDP)."
    )
    p.add_argument(
        "--model",
        choices=("tiny", "siat_30m"),
        default="tiny",
        help="Model preset (siat_30m keeps production architecture).",
    )
    p.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Training precision (default: fp32). bf16 uses autocast, no GradScaler.",
    )
    p.add_argument("--train-data", type=str, default=None)
    p.add_argument("--val-data", type=str, default=None)
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument(
        "--synthetic-dir",
        type=str,
        default=None,
        help="Write synthetic train/val bins here and use them.",
    )
    p.add_argument("--sequence-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--checkpoint-step", type=int, default=None)
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/smoke",
    )
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--val-interval", type=int, default=None)
    p.add_argument(
        "--jsonl-path",
        type=str,
        default="logs/smoke_train.jsonl",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=SMOKE_SEED)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    preset = args.model

    # Defaults differ by preset (smoke overrides only; production config untouched).
    if preset == "tiny":
        seq_default = 32
        batch_default = 2
        max_steps_default = 8
        warmup_default = 2
        train_tok, val_tok = 4096, 1024
    else:
        seq_default = 128
        batch_default = 1
        max_steps_default = 50
        warmup_default = 5
        train_tok, val_tok = 32_768, 4096

    sequence_length = (
        args.sequence_length if args.sequence_length is not None else seq_default
    )
    batch_size = args.batch_size if args.batch_size is not None else batch_default
    max_steps = args.max_steps if args.max_steps is not None else max_steps_default
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else warmup_default
    )

    if args.synthetic_dir:
        model_cfg = make_smoke_model_config(preset)
        meta = write_synthetic_bins(
            args.synthetic_dir,
            vocab_size=model_cfg.vocab_size,
            train_tokens=train_tok,
            val_tokens=val_tok,
            seed=args.seed,
        )
        train_data = Path(args.synthetic_dir) / meta["train_bin"]
        val_data = Path(args.synthetic_dir) / meta["val_bin"]
        metadata = Path(args.synthetic_dir) / "metadata.json"
        print(f"Wrote synthetic bins under {args.synthetic_dir}")
    else:
        if not args.train_data or not args.val_data:
            raise SystemExit(
                "Provide --train-data and --val-data, or --synthetic-dir."
            )
        train_data = Path(args.train_data)
        val_data = Path(args.val_data)
        metadata = Path(args.metadata) if args.metadata else None

    jsonl = args.jsonl_path
    if jsonl:
        Path(jsonl).parent.mkdir(parents=True, exist_ok=True)
        # Fresh file for this smoke run.
        Path(jsonl).write_text("", encoding="utf-8")

    try:
        result = run_fp32_smoke(
            preset=preset,
            train_data=train_data,
            val_data=val_data,
            metadata=metadata,
            sequence_length=sequence_length,
            batch_size=batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            checkpoint_step=args.checkpoint_step,
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            jsonl_path=jsonl,
            device=args.device,
            learning_rate=args.learning_rate,
            seed=args.seed,
            verbose=True,
            precision=args.precision,
        )
    except SmokeFailure as e:
        print("Siat Pretraining Smoke Test: FAILED")
        print(f"Failure Category: {e.category}")
        print(str(e))
        raise SystemExit(1) from e

    label = "BF16" if args.precision == "bf16" else "FP32"
    print(f"Siat {label} Pretraining Smoke Test: PASSED")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
