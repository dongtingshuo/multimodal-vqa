from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def sanitize_run_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name.strip())
    return "-".join(part for part in safe.split("-") if part) or "run"


def create_run_directory(base_dir: str | Path, run_name: str | None = None) -> Path:
    suffix = sanitize_run_name(run_name) if run_name else "vqa"
    output_dir = Path(base_dir) / f"{compact_timestamp()}-{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_run_metadata(config_path: str | Path, device: torch.device) -> dict[str, Any]:
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "started_at": utc_now(),
        "finished_at": None,
        "config_path": str(config_path),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": str(device),
        "cuda_device_name": cuda_name,
        "versions": {
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "transformers": _package_version("transformers"),
            "wandb": _package_version("wandb"),
        },
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_run_summary(path: str | Path, history: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    best_row = None
    for row in history:
        if row.get("best_checkpoint"):
            best_row = row
    if best_row is None and history:
        best_row = max(history, key=lambda row: float(row.get("val_vqa_score", 0.0)))
    write_json(
        path,
        {
            "best_epoch": None if best_row is None else best_row.get("epoch"),
            "best_metrics": best_row or {},
            "total_epochs": len(history),
            "total_seconds": metadata.get("total_seconds"),
            "artifacts": metadata.get("artifacts", {}),
            "wandb": metadata.get("wandb", {}),
        },
    )


def write_training_history(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_curves(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cache_dir = Path("outputs/cache/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_vqa_score"] for row in rows], marker="o", label="train VQA")
    axes[1].plot(epochs, [row["val_vqa_score"] for row in rows], marker="o", label="validation VQA")
    axes[1].plot(
        epochs,
        [row["val_top5_vqa_score"] for row in rows],
        marker="o",
        label="validation Top-5 VQA",
    )
    axes[1].set_title("VQA Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(epochs, [row["learning_rate"] for row in rows], marker="o")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


class WandbTracker:
    def __init__(self, enabled: bool = False, run=None, reason: str | None = None) -> None:
        self.enabled = enabled
        self.run = run
        self.reason = reason

    @property
    def url(self) -> str | None:
        return getattr(self.run, "url", None) if self.run is not None else None

    @property
    def run_id(self) -> str | None:
        return getattr(self.run, "id", None) if self.run is not None else None

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log(payload, step=step)

    def log_artifact_file(self, path: str | Path, name: str, artifact_type: str) -> None:
        if not self.enabled or self.run is None:
            return
        if not Path(path).is_file():
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def init_wandb_tracker(
    tracking_cfg: dict[str, Any],
    config: dict[str, Any],
    run_name: str | None,
    resume_run_id: str | None = None,
) -> WandbTracker:
    wandb_cfg = tracking_cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled and os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and os.environ.get("WANDB_API_KEY"):
        enabled = True
    if not enabled:
        return WandbTracker(enabled=False, reason="disabled")
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        return WandbTracker(enabled=False, reason="WANDB_API_KEY is not set")
    try:
        import wandb
    except ImportError:
        return WandbTracker(enabled=False, reason="wandb is not installed")

    entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY") or None
    project = os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "multimodal-vqa")
    tags = list(wandb_cfg.get("tags") or [])
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        config=config,
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
    )
    return WandbTracker(enabled=True, run=run)
