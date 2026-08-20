"""Print measured parameter counts for Siat presets.

Usage (from repo root)::

    python -m scripts.count_params
"""

from __future__ import annotations

from config import ModelConfig
from model.model import SiatForCausalLM, analytical_param_count


def main() -> None:
    for name, factory in (
        ("tiny", ModelConfig.tiny),
        ("siat_30m", ModelConfig.siat_30m),
    ):
        config = factory()
        model = SiatForCausalLM(config)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        expected = analytical_param_count(config)
        print(f"Siat parameter count ({name})")
        print("-" * 40)
        print(f"Total parameters:      {total:,}")
        print(f"Trainable parameters:  {trainable:,}")
        print(f"Parameters (M):        {total / 1e6:.4f}")
        print(f"Analytical expected:   {expected:,}")
        print(f"Difference:            {total - expected}")
        print(f"tie_embeddings:        {config.tie_embeddings}")
        print()


if __name__ == "__main__":
    main()
