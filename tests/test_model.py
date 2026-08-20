"""Tests for SiatForCausalLM full language model."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import ModelConfig
from model.model import SiatForCausalLM, analytical_param_count


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _trainable_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_construct_tiny():
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    assert model.n_layers == config.n_layers
    assert len(model.layers) == config.n_layers
    assert model.vocab_size == config.vocab_size


def test_construct_siat_30m():
    config = ModelConfig.siat_30m()
    model = SiatForCausalLM(config)
    assert len(model.layers) == 6
    assert model.d_model == 512
    assert model.vocab_size == 32000


def test_invalid_input_rank():
    model = SiatForCausalLM(ModelConfig.tiny())
    with pytest.raises(ValueError, match="rank 2"):
        model(torch.randint(0, 100, (2, 8, 1)))


def test_sequence_too_long():
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len + 1))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids)


def test_tiny_forward_logits_shape_dtype():
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_siat_30m_small_forward():
    config = ModelConfig.siat_30m()
    model = SiatForCausalLM(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, 8))
    logits = model(ids)
    assert logits.shape == (1, 8, 32000)
    assert torch.isfinite(logits).all()


def test_final_rmsnorm_and_lm_head_bias():
    model = SiatForCausalLM(ModelConfig.tiny())
    assert hasattr(model, "norm")
    assert model.lm_head.bias is None


def test_layer_parameter_independence():
    config = replace(ModelConfig.tiny(), n_layers=2)
    model = SiatForCausalLM(config)
    w0 = model.layers[0].attention.qkv.q_proj.weight
    w1 = model.layers[1].attention.qkv.q_proj.weight
    assert w0 is not w1
    assert w0.data_ptr() != w1.data_ptr()
    assert model.layers[0].attn_norm.weight is not model.layers[1].attn_norm.weight
    assert model.layers[0].ffn.gate_proj.weight is not model.layers[1].ffn.gate_proj.weight


def test_tied_embedding_identity():
    config = ModelConfig.tiny()
    assert config.tie_embeddings is True
    model = SiatForCausalLM(config)
    assert model.lm_head.weight is model.embed.weight
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_untied_embedding_independence():
    config = replace(ModelConfig.tiny(), tie_embeddings=False)
    model = SiatForCausalLM(config)
    assert model.lm_head.weight is not model.embed.weight
    assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()


def test_tied_parameter_count_siat_30m():
    config = ModelConfig.siat_30m()
    model = SiatForCausalLM(config)
    actual = _param_count(model)
    expected = analytical_param_count(config)
    assert expected == 36_837_888
    assert actual == expected
    assert _trainable_count(model) == actual


def test_untied_parameter_count_siat_30m():
    config = replace(ModelConfig.siat_30m(), tie_embeddings=False)
    model = SiatForCausalLM(config)
    actual = _param_count(model)
    expected = analytical_param_count(config)
    assert expected == 36_837_888 + 32_000 * 512
    assert expected == 53_221_888
    assert actual == expected


def test_tiny_analytical_matches_actual():
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    assert _param_count(model) == analytical_param_count(config)


def test_deterministic_eval():
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        a = model(ids)
        b = model(ids)
    assert torch.allclose(a, b, atol=1e-6)


def test_causal_invariance_token_ids():
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    model.eval()
    torch.manual_seed(0)
    a = torch.randint(0, config.vocab_size, (2, 4))
    b = a.clone()
    b[:, 2:] = torch.randint(0, config.vocab_size, (2, 2))
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    assert torch.allclose(la[:, 0, :], lb[:, 0, :], atol=1e-5)
    assert torch.allclose(la[:, 1, :], lb[:, 1, :], atol=1e-5)
    assert not torch.allclose(la[:, 3, :], lb[:, 3, :], atol=1e-5)


def test_full_gradient_flow():
    # n_layers=2 tiny: first / "middle"=last of small stack / last covered
    config = ModelConfig.tiny()
    model = SiatForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(ids)
    loss = (logits * torch.randn_like(logits)).sum()
    loss.backward()

    assert model.embed.weight.grad is not None
    assert model.layers[0].attention.qkv.q_proj.weight.grad is not None
    assert model.layers[0].ffn.gate_proj.weight.grad is not None
    assert model.layers[1].attention.qkv.q_proj.weight.grad is not None
    assert model.layers[1].ffn.down_proj.weight.grad is not None
    assert model.norm.weight.grad is not None
    # Tied: lm_head.weight is embed.weight → same grad tensor
    assert model.lm_head.weight.grad is model.embed.weight.grad
    assert torch.isfinite(model.embed.weight.grad).all()
    assert torch.isfinite(model.norm.weight.grad).all()
    assert not torch.isnan(model.layers[0].ffn.up_proj.weight.grad).any()


def test_untied_gradients_independent():
    config = replace(ModelConfig.tiny(), tie_embeddings=False)
    model = SiatForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(ids)
    (logits * torch.randn_like(logits)).sum().backward()
    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.embed.weight.grad.data_ptr() != model.lm_head.weight.grad.data_ptr()


def test_state_dict_and_roundtrip():
    config = ModelConfig.tiny()
    model_a = SiatForCausalLM(config)
    model_a.eval()
    state = model_a.state_dict()
    assert "embed.embedding.weight" in state
    # Tied: lm_head.weight still appears as its own key in state_dict
    assert "lm_head.weight" in state
    assert "norm.weight" in state

    model_b = SiatForCausalLM(config)
    model_b.load_state_dict(state)
    model_b.eval()
    ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        assert torch.allclose(model_a(ids), model_b(ids), atol=1e-6)

    # After load, tying is preserved by construction on new model;
    # verify shared storage still holds on model_a.
    assert model_a.lm_head.weight is model_a.embed.weight


def test_tied_state_dict_same_storage():
    model = SiatForCausalLM(ModelConfig.tiny())
    state = model.state_dict()
    # Both keys refer to the same underlying tensor when tied.
    assert torch.equal(state["embed.embedding.weight"], state["lm_head.weight"])
    assert (
        state["embed.embedding.weight"].data_ptr()
        == state["lm_head.weight"].data_ptr()
    )
