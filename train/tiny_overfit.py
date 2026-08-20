"""Tiny overfit smoke: verify Dataset → Model → Loss → AdamW can drive loss down.

Usage (from repo root)::

    python -m train.tiny_overfit

Uses a test-only small ``ModelConfig`` (does not mutate ``ModelConfig.tiny()``).
Synthetic fixed token sequences are repeated — no external corpus download.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from config import ModelConfig
from model.model import SiatForCausalLM
from train.loss import causal_lm_loss

# Fixed hyperparameters for this smoke test only.
OVERFIT_LR = 3e-3
OVERFIT_WEIGHT_DECAY = 0.0
OVERFIT_STEPS_DEFAULT = 80
OVERFIT_LOG_EVERY = 10
OVERFIT_SEED = 42


def make_overfit_config() -> ModelConfig:
    """Ultra-small config for fast CPU overfit (not production tiny/30M)."""
    return ModelConfig(
        model_name="Siat",
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
        tie_embeddings=True,
    )


def build_synthetic_batch(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    seed: int = OVERFIT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed ``input_ids`` / already-shifted ``labels`` (Dataset contract).

    For each sample, tokens are a short repeating pattern; labels are tokens[1:].
    """
    g = torch.Generator().manual_seed(seed)
    # Full stream length = seq_len + 1 so we can form input[:-] / labels[1:]
    streams = torch.randint(
        0, vocab_size, (batch_size, seq_len + 1), generator=g
    )
    # Make patterns more memorable: clamp diversity a bit by repeating a motif
    motif = torch.arange(seq_len + 1) % max(vocab_size // 4, 2)
    streams = (streams % 8 + motif.unsqueeze(0)) % vocab_size
    input_ids = streams[:, :-1].contiguous()
    labels = streams[:, 1:].contiguous().long()
    return input_ids, labels


def next_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


@dataclass
class OverfitResult:
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    steps: int
    lr: float
    device: str

    @property
    def reduction_ratio(self) -> float:
        if self.initial_loss <= 0:
            return 0.0
        return 1.0 - (self.final_loss / self.initial_loss)

    @property
    def reduction_pct(self) -> float:
        return 100.0 * self.reduction_ratio


def run_tiny_overfit(
    *,
    steps: int = OVERFIT_STEPS_DEFAULT,
    lr: float = OVERFIT_LR,
    seed: int = OVERFIT_SEED,
    seq_len: int = 16,
    batch_size: int = 4,
    device: str | None = None,
    log_every: int = OVERFIT_LOG_EVERY,
    verbose: bool = True,
) -> OverfitResult:
    """Train on one repeated synthetic batch; return initial/final metrics."""
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = make_overfit_config()
    model = SiatForCausalLM(config).to(device)
    model.train()

    input_ids, labels = build_synthetic_batch(
        config.vocab_size, seq_len, batch_size, seed=seed
    )
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=OVERFIT_WEIGHT_DECAY,
    )

    with torch.no_grad():
        init_logits = model(input_ids)
        initial_loss = causal_lm_loss(init_logits, labels).item()
        initial_acc = next_token_accuracy(init_logits, labels)

    if verbose:
        print("Tiny Overfit Sanity Check")
        print("-" * 40)
        print(f"device={device}  steps={steps}  lr={lr}  wd={OVERFIT_WEIGHT_DECAY}")
        print(
            f"vocab={config.vocab_size} d_model={config.d_model} "
            f"n_layers={config.n_layers} n_heads={config.n_heads} "
            f"ffn_dim={config.ffn_dim} seq_len={seq_len} batch={batch_size}"
        )
        print(f"step   0 | loss {initial_loss:.4f} | acc {initial_acc:.4f}")

    final_loss = initial_loss
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if verbose and (step % log_every == 0 or step == steps):
            with torch.no_grad():
                acc = next_token_accuracy(logits, labels)
            print(f"step {step:3d} | loss {final_loss:.4f} | acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        eval_logits = model(input_ids)
        eval_loss = causal_lm_loss(eval_logits, labels).item()
        final_acc = next_token_accuracy(eval_logits, labels)

    if verbose:
        reduction = 100.0 * (1.0 - eval_loss / initial_loss) if initial_loss else 0.0
        print("-" * 40)
        print(f"Initial Loss:     {initial_loss:.6f}")
        print(f"Final Loss:       {eval_loss:.6f}")
        print(f"Reduction:        {reduction:.2f}%")
        print(f"Initial Accuracy: {initial_acc:.4f}")
        print(f"Final Accuracy:   {final_acc:.4f}")
        # A few token ID comparisons from last batch
        preds = eval_logits.argmax(dim=-1)
        print(
            f"sample targets[:8]={labels[0, :8].tolist()} "
            f"preds[:8]={preds[0, :8].tolist()}"
        )

    return OverfitResult(
        initial_loss=initial_loss,
        final_loss=eval_loss,
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        steps=steps,
        lr=lr,
        device=device,
    )


def main() -> None:
    result = run_tiny_overfit()
    if result.final_loss >= result.initial_loss * 0.5:
        raise SystemExit(
            f"Tiny Overfit FAILED: final={result.final_loss:.4f} "
            f"not < 0.5 * initial={result.initial_loss:.4f}"
        )
    if not math.isfinite(result.final_loss):
        raise SystemExit("Tiny Overfit FAILED: non-finite final loss")
    print("Tiny Overfit Test: PASSED")


if __name__ == "__main__":
    main()
