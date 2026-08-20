"""Core training loop utilities for Siat.

AdamW (decay/no_decay), warmup + cosine LR, gradient accumulation, clipping,
validation, checkpoint save/resume, basic console/JSONL logging, and optional
BF16 autocast (no GradScaler / no FP16 / no distributed).
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from train.loss import causal_lm_loss

# Architecture fields checked on resume (TrainConfig may differ intentionally).
_ARCH_FIELDS = (
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "ffn_dim",
    "max_seq_len",
    "tie_embeddings",
)


def is_bf16_supported(device: str | torch.device) -> bool:
    """Return True if BF16 autocast training is safely supported on ``device``."""
    device = torch.device(device)
    if device.type == "cuda":
        return bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
    if device.type == "cpu":
        checker = getattr(torch.cpu, "is_bf16_supported", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False
    return False


def require_bf16_support(device: str | torch.device) -> None:
    """Raise RuntimeError if BF16 was requested on an unsupported device."""
    device = torch.device(device)
    if is_bf16_supported(device):
        return
    raise RuntimeError(
        "BF16 precision was requested but is not supported on the selected "
        f"device ({device}). Use precision='fp32' or a BF16-capable device."
    )


def get_learning_rate(step: int, config: TrainConfig) -> float:
    """Learning rate for optimizer step ``step`` (0-based).

    Convention
    ----------
    * Warmup (``0 <= step < warmup_steps``)::

          lr = peak * (step + 1) / warmup_steps

      so step 0 is ``peak / warmup_steps`` and step ``warmup_steps - 1`` is peak.

    * Cosine (``warmup_steps <= step < max_steps``)::

          progress = (step - warmup_steps) / (max_steps - warmup_steps)
          lr = min_lr + 0.5 * (1 + cos(pi * progress)) * (peak - min_lr)

    * ``step >= max_steps`` → ``min_learning_rate`` (clamp).
    """
    peak = config.learning_rate
    min_lr = config.min_learning_rate
    warmup = config.warmup_steps
    max_steps = config.max_steps

    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}.")
    if step >= max_steps:
        return float(min_lr)
    if step < warmup:
        return float(peak * (step + 1) / warmup)

    denom = max_steps - warmup
    progress = (step - warmup) / denom
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + cosine * (peak - min_lr))


def build_adamw_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split parameters by tensor rank; dedupe by ``id`` (weight tying safe)."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _grad_l2_norm(parameters: Iterator[nn.Parameter] | list[nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm(2).item() ** 2)
    return math.sqrt(total)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class SiatTrainer:
    """Trainer with FP32 default and optional BF16 autocast.

    Master parameters and AdamW state remain FP32. ``TrainConfig.precision``
    selects ``fp32`` (no autocast) or ``bf16`` (autocast, no GradScaler).

    ``TrainConfig.max_steps`` counts **optimizer updates**, not micro-batches.
    Effective batch (single-device) = ``batch_size * gradient_accumulation_steps``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        device: str | torch.device = "cpu",
        *,
        model_config: ModelConfig | None = None,
        precision: str | None = None,
    ) -> None:
        self.model = model
        self.config = train_config
        self.model_config = model_config
        self.device = torch.device(device)

        # Keyword override wins; otherwise TrainConfig.precision is source of truth.
        self.precision = (
            train_config.precision if precision is None else str(precision)
        )
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(
                f"precision must be 'fp32' or 'bf16', got {self.precision!r}."
            )
        if self.precision == "bf16":
            require_bf16_support(self.device)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            build_adamw_param_groups(self.model, train_config.weight_decay),
            lr=train_config.learning_rate,
        )

        self.optimizer_step = 0
        self.micro_step = 0
        self.tokens_processed = 0
        self._window_tokens = 0
        self._window_t0 = time.perf_counter()
        self.last_val_loss: float | None = None

    def _autocast_context(self):
        """BF16 autocast for forward; FP32 uses a no-op context."""
        if self.precision == "bf16":
            return torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
            )
        return contextlib.nullcontext()

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _move_batch(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        return input_ids, labels

    def _infinite_batches(self, dataloader: DataLoader) -> Iterator[Any]:
        while True:
            for batch in dataloader:
                yield batch

    def validate(
        self,
        val_dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Run eval-mode validation; restore prior train/eval mode afterward."""
        if val_dataloader is None:
            raise ValueError("val_dataloader must not be None.")

        was_training = self.model.training
        self.model.eval()

        total_loss_tokens = 0.0
        total_tokens = 0
        n_batches = 0

        try:
            with torch.no_grad():
                for batch in val_dataloader:
                    if max_batches is not None and n_batches >= max_batches:
                        break
                    input_ids, labels = self._move_batch(batch)
                    with self._autocast_context():
                        logits = self.model(input_ids)
                        loss = causal_lm_loss(logits, labels)
                    n_tok = int(labels.numel())
                    total_loss_tokens += float(loss.item()) * n_tok
                    total_tokens += n_tok
                    n_batches += 1
        finally:
            self.model.train(was_training)

        if n_batches == 0 or total_tokens == 0:
            raise ValueError(
                "Validation DataLoader produced zero batches/tokens "
                "(empty loader or max_batches=0)."
            )

        val_loss = total_loss_tokens / total_tokens
        if val_loss < 20.0:
            perplexity = float(math.exp(val_loss))
        else:
            perplexity = float("inf")

        self.last_val_loss = val_loss
        return {"val_loss": val_loss, "perplexity": perplexity}

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model/optimizer/training state at an accumulation boundary."""
        path = Path(path)
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_step": self.optimizer_step,
            "micro_step": self.micro_step,
            "tokens_processed": self.tokens_processed,
            "train_config": asdict(self.config),
            "precision": self.precision,
            "rng_state": torch.get_rng_state(),
        }
        if self.model_config is not None:
            payload["model_config"] = asdict(self.model_config)
        if self.last_val_loss is not None:
            payload["last_val_loss"] = self.last_val_loss
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        _atomic_torch_save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, counters, and RNG from ``path``."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if "model_config" in ckpt and self.model_config is not None:
            saved = ckpt["model_config"]
            for key in _ARCH_FIELDS:
                if key not in saved:
                    continue
                current = getattr(self.model_config, key)
                if saved[key] != current:
                    raise ValueError(
                        f"Checkpoint model_config.{key}={saved[key]!r} "
                        f"does not match current {current!r}."
                    )

        ckpt_precision = ckpt.get("precision")
        if ckpt_precision is None and isinstance(ckpt.get("train_config"), dict):
            ckpt_precision = ckpt["train_config"].get("precision")
        if (
            ckpt_precision is not None
            and ckpt_precision != self.precision
        ):
            warnings.warn(
                f"Checkpoint precision={ckpt_precision!r} differs from "
                f"current precision={self.precision!r}; continuing with "
                f"current runtime precision.",
                UserWarning,
                stacklevel=2,
            )

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.optimizer_step = int(ckpt["optimizer_step"])
        self.micro_step = int(ckpt.get("micro_step", 0))
        self.tokens_processed = int(ckpt.get("tokens_processed", 0))
        if "last_val_loss" in ckpt:
            self.last_val_loss = float(ckpt["last_val_loss"])

        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        if (
            "cuda_rng_state_all" in ckpt
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

        # Ensure optimizer state tensors live on the trainer device.
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def train(
        self,
        dataloader: DataLoader,
        *,
        log_interval: int = 10,
        max_steps: int | None = None,
        val_dataloader: DataLoader | None = None,
        val_interval: int = 0,
        val_max_batches: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 0,
        jsonl_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Run until ``optimizer_step`` reaches ``max_steps``.

        ``val_interval`` / ``checkpoint_interval`` are in **optimizer steps**;
        ``0`` disables that feature. Checkpoints are written only after a full
        accumulation cycle (post ``optimizer.step``).
        """
        target_steps = (
            self.config.max_steps if max_steps is None else max_steps
        )
        if target_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {target_steps}.")
        if val_interval < 0:
            raise ValueError(f"val_interval must be >= 0, got {val_interval}.")
        if checkpoint_interval < 0:
            raise ValueError(
                f"checkpoint_interval must be >= 0, got {checkpoint_interval}."
            )

        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        accum = self.config.gradient_accumulation_steps
        self.model.train()
        history: list[dict[str, float]] = []
        batch_iter = self._infinite_batches(dataloader)
        self._window_tokens = 0
        self._window_t0 = time.perf_counter()

        while self.optimizer_step < target_steps:
            self.optimizer.zero_grad(set_to_none=True)
            raw_losses: list[float] = []
            step_tokens = 0

            for _ in range(accum):
                batch = next(batch_iter)
                input_ids, labels = self._move_batch(batch)
                with self._autocast_context():
                    logits = self.model(input_ids)
                    loss = causal_lm_loss(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at micro_step={self.micro_step}, "
                        f"optimizer_step={self.optimizer_step}: {loss.item()}"
                    )
                raw_losses.append(float(loss.item()))
                n_tok = int(input_ids.numel())
                step_tokens += n_tok
                (loss / accum).backward()
                self.micro_step += 1

            if self.config.max_grad_norm > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                )
            else:
                grad_norm = _grad_l2_norm(self.model.parameters())

            lr = get_learning_rate(self.optimizer_step, self.config)
            self._set_lr(lr)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1

            self.tokens_processed += step_tokens
            self._window_tokens += step_tokens
            mean_loss = sum(raw_losses) / len(raw_losses)

            elapsed = time.perf_counter() - self._window_t0
            tok_s = (
                self._window_tokens / elapsed if elapsed > 0 else 0.0
            )

            metrics: dict[str, float] = {
                "step": float(self.optimizer_step),
                "loss": mean_loss,
                "train_loss": mean_loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "tokens_processed": float(self.tokens_processed),
                "tokens_per_second": tok_s,
            }

            if log_interval > 0 and (
                self.optimizer_step % log_interval == 0
                or self.optimizer_step == 1
            ):
                print(
                    f"step {self.optimizer_step} | train_loss {mean_loss:.4f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm:.4f} | "
                    f"tok/s {tok_s:.0f} | tokens {self.tokens_processed}"
                )
                if jsonl_path is not None:
                    rec = dict(metrics)
                    rec["precision"] = self.precision
                    _append_jsonl(Path(jsonl_path), rec)
                # Reset throughput window after a logged interval.
                self._window_tokens = 0
                self._window_t0 = time.perf_counter()

            if (
                val_dataloader is not None
                and val_interval > 0
                and self.optimizer_step % val_interval == 0
            ):
                val_metrics = self.validate(
                    val_dataloader, max_batches=val_max_batches
                )
                metrics["val_loss"] = val_metrics["val_loss"]
                metrics["perplexity"] = val_metrics["perplexity"]
                print(
                    f"step {self.optimizer_step} | "
                    f"val_loss {val_metrics['val_loss']:.4f} | "
                    f"ppl {val_metrics['perplexity']:.4f}"
                )
                if jsonl_path is not None:
                    _append_jsonl(
                        Path(jsonl_path),
                        {
                            "step": self.optimizer_step,
                            "val_loss": val_metrics["val_loss"],
                            "perplexity": val_metrics["perplexity"],
                            "precision": self.precision,
                        },
                    )
                self.model.train()

            if (
                ckpt_dir is not None
                and checkpoint_interval > 0
                and self.optimizer_step % checkpoint_interval == 0
            ):
                step_path = ckpt_dir / f"step_{self.optimizer_step:06d}.pt"
                self.save_checkpoint(step_path)
                self.save_checkpoint(ckpt_dir / "latest.pt")
                print(f"saved checkpoint {step_path}")

            history.append(metrics)

        return history
