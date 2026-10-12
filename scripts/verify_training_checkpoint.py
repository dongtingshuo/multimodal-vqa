"""Inspect a training checkpoint in a short-lived CPU process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa_project.engine import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--min-updates", type=int, default=0)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if not checkpoint.get("model_state"):
        raise ValueError("Checkpoint has no model parameters.")
    state = checkpoint.get("training_state", {})
    updates = int(state.get("global_step", 0))
    if updates < args.min_updates:
        raise ValueError(f"Checkpoint has {updates} successful updates; required {args.min_updates}.")
    if args.min_updates > 0:
        for key in ("optimizer_state", "scaler_state"):
            if not isinstance(checkpoint.get(key), dict):
                raise ValueError(f"Training checkpoint is missing {key}.")
    print(json.dumps({
        "global_step": updates,
        "epoch": int(checkpoint.get("epoch", 0)),
        "epoch_complete": bool(state.get("epoch_complete", True)),
        "format_version": checkpoint.get("format_version"),
    }))


if __name__ == "__main__":
    main()
