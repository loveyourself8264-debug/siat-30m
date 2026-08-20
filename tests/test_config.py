"""Tests for Siat configuration validation and presets."""

import pytest

from config import Config, ModelConfig, TrainConfig


def test_default_config_creation():
    model = ModelConfig()
    train = TrainConfig()
    cfg = Config()
    assert model.model_name == "Siat"
    assert cfg.model.d_model == model.d_model
    assert cfg.train.batch_size == train.batch_size


def test_tiny_config():
    model = ModelConfig.tiny()
    assert model.model_name == "Siat"
    assert model.vocab_size == 8000
    assert model.d_model == 128
    assert model.n_layers == 2
    assert model.n_heads == 4
    assert model.ffn_dim == 512
    assert model.max_seq_len == 256
    assert model.head_dim == 32

    cfg = Config.tiny()
    assert cfg.model.d_model == 128
    assert cfg.train.max_steps == 100


def test_siat_30m_config():
    model = ModelConfig.siat_30m()
    assert model.model_name == "Siat"
    assert model.vocab_size == 32000
    assert model.d_model == 512
    assert model.n_layers == 6
    assert model.n_heads == 8
    assert model.ffn_dim == 1536
    assert model.max_seq_len == 1024
    assert model.head_dim == 64

    cfg = Config.siat_30m()
    assert cfg.model.model_name == "Siat"
    assert cfg.train.max_steps == 100_000


def test_head_dim_calculation():
    model = ModelConfig(d_model=256, n_heads=8)
    assert model.head_dim == 32


def test_invalid_d_model_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=130, n_heads=4)


def test_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.5)
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=-0.1)


def test_invalid_non_positive_model_fields():
    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError, match="n_layers"):
        ModelConfig(n_layers=-1)
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=0)


def test_invalid_rope_head_dim():
    # d_model == n_heads -> head_dim == 1 (odd), unsuitable for RoPE pairs.
    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig(d_model=4, n_heads=4)


def test_invalid_train_fields():
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=-1e-3)
    with pytest.raises(ValueError, match="weight_decay"):
        TrainConfig(weight_decay=-0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(warmup_steps=100, max_steps=50)
    with pytest.raises(ValueError, match="min_learning_rate"):
        TrainConfig(learning_rate=1e-4, min_learning_rate=1e-3)


def test_precision_default_and_validation():
    assert TrainConfig().precision == "fp32"
    assert TrainConfig(precision="bf16").precision == "bf16"
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp16")
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="auto")


def test_empty_model_name():
    with pytest.raises(ValueError, match="model_name"):
        ModelConfig(model_name="   ")
