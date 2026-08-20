"""Siat (씨앗) model and training configuration.

Provides dataclass-based configs with validation, plus tiny / Siat 30M presets.
Future options (e.g. mixed precision) can be added to Config without breaking
existing ModelConfig / TrainConfig fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters for Siat decoder-only LM."""

    model_name: str = "Siat"
    vocab_size: int = 8000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        """Per-head dimension: d_model // n_heads."""
        return self.d_model // self.n_heads

    def validate(self) -> None:
        """Raise ValueError if model settings are invalid."""
        if not self.model_name or not str(self.model_name).strip():
            raise ValueError("model_name must be a non-empty string.")

        positive_ints = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0], got {self.dropout}."
            )

        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}.")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be > 0, got {self.rms_norm_eps}.")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )

        # RoPE applies rotations to pairs of dimensions (cos/sin), so head_dim must be even.
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim (= d_model // n_heads = {self.head_dim}) must be even "
                f"for RoPE, but d_model={self.d_model}, n_heads={self.n_heads}."
            )

    @classmethod
    def tiny(cls) -> ModelConfig:
        """Small debug config for end-to-end pipeline checks before Siat 30M."""
        return cls(
            model_name="Siat",
            vocab_size=8000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            ffn_dim=512,
            max_seq_len=256,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
        )

    @classmethod
    def siat_30m(cls) -> ModelConfig:
        """Candidate config targeting ~30M parameters (exact count TBD after model impl).

        Rough rationale (not a hard parameter budget):
        - vocab_size=32000: common BPE scale for small LMs
        - d_model=512, n_heads=8 -> head_dim=64 (standard for RoPE)
        - n_layers=6: depth/width balance for ~30M class
        - ffn_dim=1536 (~3x d_model): comfortable for SwiGLU
        - max_seq_len=1024: pretraining context candidate
        - tie_embeddings=True: save embedding/lm_head parameters
        """
        return cls(
            model_name="Siat",
            vocab_size=32000,
            d_model=512,
            n_layers=6,
            n_heads=8,
            ffn_dim=1536,
            max_seq_len=1024,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
        )


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""

    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 10
    max_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    # Runtime training precision: "fp32" (default) or "bf16" (autocast; no GradScaler).
    precision: str = "fp32"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if training settings are invalid."""
        positive_ints = {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.min_learning_rate <= 0:
            raise ValueError(
                f"min_learning_rate must be > 0, got {self.min_learning_rate}."
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}.")
        if self.max_grad_norm < 0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}.")

        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps "
                f"({self.max_steps})."
            )
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                f"min_learning_rate ({self.min_learning_rate}) must be "
                f"<= learning_rate ({self.learning_rate})."
            )

        allowed_precision = {"fp32", "bf16"}
        if self.precision not in allowed_precision:
            raise ValueError(
                f"precision must be one of {sorted(allowed_precision)}, "
                f"got {self.precision!r}."
            )


@dataclass
class Config:
    """Top-level config bundling model + train settings.

    Precision lives on ``TrainConfig.precision`` (fp32 default / bf16 autocast).
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def tiny(cls) -> Config:
        """Tiny Siat + debug-friendly training defaults."""
        return cls(
            model=ModelConfig.tiny(),
            train=TrainConfig(
                batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=3e-4,
                min_learning_rate=3e-5,
                weight_decay=0.1,
                warmup_steps=10,
                max_steps=100,
                max_grad_norm=1.0,
                seed=42,
            ),
        )

    @classmethod
    def siat_30m(cls) -> Config:
        """Siat 30M candidate model + pretraining-oriented train defaults."""
        return cls(
            model=ModelConfig.siat_30m(),
            train=TrainConfig(
                batch_size=8,
                gradient_accumulation_steps=4,
                learning_rate=3e-4,
                min_learning_rate=3e-5,
                weight_decay=0.1,
                warmup_steps=1000,
                max_steps=100_000,
                max_grad_norm=1.0,
                seed=42,
            ),
        )
