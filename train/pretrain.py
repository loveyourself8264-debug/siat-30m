"""Pretraining entrypoint for Siat (validation + checkpoint + logging).

Example::

    python -m train.pretrain \\
        --train-data data/processed/train.bin \\
        --val-data data/processed/val.bin \\
        --checkpoint-dir checkpoints \\
        --precision bf16 \\
        --log-interval 10 \\
        --val-interval 500 \\
        --checkpoint-interval 1000

Resume::

    python -m train.pretrain ... --resume checkpoints/step_010000.pt

Supports ``--precision fp32`` (default) and ``bf16`` autocast.
Does not implement FP16, GradScaler, or distributed training.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from config import Config, ModelConfig, TrainConfig
from data.dataset import SiatDataset, create_dataloader
from model.model import SiatForCausalLM
from train.trainer import SiatTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Siat pretrain (fp32 / bf16 autocast, val/ckpt/logging)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to train.bin",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Optional path to val.bin",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.json (default: sibling of train.bin)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=("tiny", "siat_30m"),
        default="tiny",
        help="Preset Config",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--val-interval",
        type=int,
        default=0,
        help="Validate every N optimizer steps (0=disabled)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save checkpoint every N optimizer steps (0=disabled)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for step_XXXXXX.pt / latest.pt",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="Optional JSONL metrics file (append)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("fp32", "bf16"),
        default="fp32",
        help="fp32 (default) or bf16 autocast (no GradScaler)",
    )
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    cfg: Config = Config.tiny() if args.config == "tiny" else Config.siat_30m()
    model_cfg: ModelConfig = cfg.model
    train_cfg: TrainConfig = replace(cfg.train, precision=args.precision)

    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_bin = Path(args.train_data)
    metadata = Path(args.metadata) if args.metadata else None

    train_ds = _build_dataset(train_bin, model_cfg.max_seq_len, metadata)
    if len(train_ds) == 0:
        raise SystemExit(
            f"Train dataset empty for sequence_length={model_cfg.max_seq_len}: "
            f"{train_bin}"
        )
    train_loader = create_dataloader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = None
    if args.val_data:
        val_bin = Path(args.val_data)
        val_ds = _build_dataset(val_bin, model_cfg.max_seq_len, metadata)
        if len(val_ds) == 0:
            raise SystemExit(f"Val dataset empty: {val_bin}")
        val_loader = create_dataloader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

    model = SiatForCausalLM(model_cfg)
    trainer = SiatTrainer(
        model, train_cfg, device=device, model_config=model_cfg
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print(
        f"Siat pretrain | config={args.config} device={device} "
        f"precision={train_cfg.precision} "
        f"max_steps={args.max_steps or train_cfg.max_steps} "
        f"accum={train_cfg.gradient_accumulation_steps} "
        f"effective_batch="
        f"{train_cfg.batch_size * train_cfg.gradient_accumulation_steps} "
        f"val_interval={args.val_interval} "
        f"ckpt_interval={args.checkpoint_interval}"
    )
    trainer.train(
        train_loader,
        log_interval=args.log_interval,
        max_steps=args.max_steps,
        val_dataloader=val_loader,
        val_interval=args.val_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        jsonl_path=args.jsonl_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
