from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from vqa_project.config import load_config
from vqa_project.engine import load_checkpoint
from vqa_project.training import validate_resume_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an AutoDL ViLT continuation environment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--answer-vocab")
    parser.add_argument("--require-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU instance and a CUDA-enabled PyTorch image.")

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.answer_vocab:
        config["data"]["answer_vocab_path"] = args.answer_vocab

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    validate_resume_config(config, checkpoint)
    if int(checkpoint.get("epoch", -1)) != 2:
        raise ValueError(f"Expected an epoch-2 checkpoint, found epoch={checkpoint.get('epoch')!r}.")
    if len(checkpoint.get("idx_to_answer") or []) != 3000:
        raise ValueError("Checkpoint answer vocabulary is not Top-3000.")
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "rng_state", "training_state"):
        if key not in checkpoint:
            raise ValueError(f"Resume checkpoint is missing {key!r}.")

    if args.answer_vocab and not Path(args.answer_vocab).is_file():
        raise FileNotFoundError(f"Missing answer vocabulary: {args.answer_vocab}")
    if args.require_data:
        required = (
            "train2014",
            "val2014",
            "v2_OpenEnded_mscoco_train2014_questions.json",
            "v2_mscoco_train2014_annotations.json",
            "v2_OpenEnded_mscoco_val2014_questions.json",
            "v2_mscoco_val2014_annotations.json",
        )
        missing = [name for name in required if not (Path(config["data"]["root"]) / name).exists()]
        if missing:
            raise FileNotFoundError("Incomplete data root; missing: " + ", ".join(missing))

    free_bytes = shutil.disk_usage(checkpoint_path.parent).free
    payload = {
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "resume_epoch": int(checkpoint["epoch"]) + 1,
        "best_metric": (checkpoint.get("training_state") or {}).get("best_metric"),
        "global_step": (checkpoint.get("training_state") or {}).get("global_step"),
        "free_disk_gib": round(free_bytes / 1024**3, 2),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
