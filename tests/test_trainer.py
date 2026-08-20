"""Tests for SiatTrainer: AdamW groups, accum, clip, smoke training."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, TrainConfig
from model.model import SiatForCausalLM
from train.trainer import SiatTrainer, build_adamw_param_groups, get_learning_rate


def _tiny_model_config() -> ModelConfig:
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


def _train_cfg(**kwargs) -> TrainConfig:
    defaults = dict(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=5,
        max_steps=20,
        max_grad_norm=1.0,
        seed=42,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def _batch_loader(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    ds = TensorDataset(input_ids, labels)

    def collate(samples):
        ids = torch.stack([s[0] for s in samples], dim=0)
        labs = torch.stack([s[1] for s in samples], dim=0)
        return {"input_ids": ids, "labels": labs}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def test_adamw_param_coverage_and_no_duplicates():
    model = SiatForCausalLM(_tiny_model_config())
    groups = build_adamw_param_groups(model, weight_decay=0.1)
    opt_ids = [id(p) for g in groups for p in g["params"]]
    assert len(opt_ids) == len(set(opt_ids))

    model_ids = {
        id(p) for p in model.parameters() if p.requires_grad
    }
    # Tied lm_head.weight is the same Parameter as embed — counted once.
    assert set(opt_ids) == model_ids
    assert id(model.lm_head.weight) == id(model.embed.weight)
    assert opt_ids.count(id(model.embed.weight)) == 1


def test_decay_no_decay_grouping():
    model = SiatForCausalLM(_tiny_model_config())
    wd = 0.1
    groups = build_adamw_param_groups(model, weight_decay=wd)
    decay = {id(p) for p in groups[0]["params"]}
    no_decay = {id(p) for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == wd
    assert groups[1]["weight_decay"] == 0.0

    # Linear / embedding matrices → decay
    assert id(model.layers[0].attention.qkv.q_proj.weight) in decay
    assert id(model.embed.weight) in decay
    # RMSNorm 1D → no_decay
    assert id(model.norm.weight) in no_decay
    assert id(model.layers[0].attn_norm.weight) in no_decay


def test_optimizer_step_only_after_full_accum():
    torch.manual_seed(0)
    model = SiatForCausalLM(_tiny_model_config())
    train_cfg = _train_cfg(gradient_accumulation_steps=4, max_steps=2, warmup_steps=1)
    trainer = SiatTrainer(model, train_cfg, device="cpu")

    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    # Manually run 3 micros — params must not change yet
    trainer.optimizer.zero_grad(set_to_none=True)
    ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    from train.loss import causal_lm_loss

    for _ in range(3):
        loss = causal_lm_loss(model(ids), labels)
        (loss / 4).backward()
        trainer.micro_step += 1

    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, before[n])

    # 4th micro + clip + step → params change
    loss = causal_lm_loss(model(ids), labels)
    (loss / 4).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
    trainer._set_lr(get_learning_rate(0, train_cfg))
    trainer.optimizer.step()

    changed = False
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.equal(p, before[n]):
            changed = True
            break
    assert changed


def test_grad_accum_matches_large_batch():
    """accum=2 with micro B=2 ≈ single step with B=4 (dropout=0, same order)."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    base = SiatForCausalLM(config)
    state = copy.deepcopy(base.state_dict())

    # Shared 4 samples
    g = torch.Generator().manual_seed(1)
    all_ids = torch.randint(0, 64, (4, 8), generator=g)
    all_labels = torch.randint(0, 64, (4, 8), generator=g)

    def one_opt_step(model, loader, accum, lr=1e-3):
        train_cfg = _train_cfg(
            gradient_accumulation_steps=accum,
            max_steps=2,
            warmup_steps=1,
            learning_rate=lr,
            min_learning_rate=lr,
            weight_decay=0.0,
            max_grad_norm=0.0,
        )
        trainer = SiatTrainer(model, train_cfg, device="cpu")
        trainer.train(loader, log_interval=0, max_steps=1)
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    model_a = SiatForCausalLM(config)
    model_a.load_state_dict(state)
    loader_a = _batch_loader(all_ids, all_labels, batch_size=4)
    params_a = one_opt_step(model_a, loader_a, accum=1)

    model_b = SiatForCausalLM(config)
    model_b.load_state_dict(state)
    loader_b = _batch_loader(all_ids, all_labels, batch_size=2)
    params_b = one_opt_step(model_b, loader_b, accum=2)

    for name in params_a:
        assert torch.allclose(params_a[name], params_b[name], atol=1e-5, rtol=1e-4), name


def test_gradient_clipping_reduces_norm():
    torch.manual_seed(0)
    model = SiatForCausalLM(_tiny_model_config())
    train_cfg = _train_cfg(max_grad_norm=0.5, max_steps=2, warmup_steps=1)
    trainer = SiatTrainer(model, train_cfg, device="cpu")
    ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    from train.loss import causal_lm_loss

    trainer.optimizer.zero_grad(set_to_none=True)
    # Amplify grads artificially after backward
    loss = causal_lm_loss(model(ids), labels)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(100.0)

    pre = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
    )
    assert pre > train_cfg.max_grad_norm
    # After clip_grad_norm_, norms should be <= max_grad_norm
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    post = total ** 0.5
    assert post <= train_cfg.max_grad_norm + 1e-5


def test_clipping_disabled_when_max_grad_norm_zero():
    model = SiatForCausalLM(_tiny_model_config())
    train_cfg = _train_cfg(max_grad_norm=0.0, max_steps=2, warmup_steps=1)
    ids = torch.randint(0, 64, (8, 8))
    labels = torch.randint(0, 64, (8, 8))
    loader = _batch_loader(ids, labels, batch_size=2)
    trainer = SiatTrainer(model, train_cfg, device="cpu")
    hist = trainer.train(loader, log_interval=0, max_steps=2)
    assert all(m["grad_norm"] >= 0 for m in hist)


def test_tiny_training_smoke():
    torch.manual_seed(42)
    model = SiatForCausalLM(_tiny_model_config())
    train_cfg = _train_cfg(
        max_steps=10,
        warmup_steps=3,
        gradient_accumulation_steps=2,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
    )
    ids = torch.randint(0, 64, (16, 8))
    labels = torch.randint(0, 64, (16, 8))
    loader = _batch_loader(ids, labels, batch_size=2)

    before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    trainer = SiatTrainer(model, train_cfg, device="cpu")
    hist = trainer.train(loader, log_interval=0, max_steps=10)

    assert len(hist) == 10
    assert trainer.optimizer_step == 10
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in hist)
    assert hist[0]["learning_rate"] == get_learning_rate(0, train_cfg)
    assert hist[2]["learning_rate"] == get_learning_rate(2, train_cfg)
    # LR changes across warmup
    assert hist[0]["learning_rate"] < hist[2]["learning_rate"]
    after = next(p for p in model.parameters() if p.requires_grad)
    assert not torch.equal(before, after)
