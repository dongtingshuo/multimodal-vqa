from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "root": "data/vqa",
        "train_split": "train",
        "val_split": "val",
        "answer_vocab_path": "data/answer_vocab.json",
        "answer_vocab_size": 1000,
        "max_train_samples": 5000,
        "max_val_samples": 1000,
        "image_size": 224,
        "max_question_length": 32,
        "processor_name": None,
        "num_workers": 0,
        "augmentation": {
            "enabled": False,
            "random_resized_crop": True,
            "horizontal_flip": 0.5,
            "color_jitter": 0.0,
        },
    },
    "model": {
        "name": "cross_attention",
        "text_model_name": "distilbert-base-uncased",
        "hidden_dim": 512,
        "num_attention_heads": 8,
        "dropout": 0.2,
        "freeze_backbones": True,
        "pretrained_cnn": True,
    },
    "train": {
        "batch_size": 8,
        "epochs": 5,
        "lr": 1e-4,
        "image_lr": 1e-5,
        "text_lr": 5e-6,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "staged_finetuning": False,
        "fixed_finetune_stage": None,
        "freeze_epochs": 2,
        "unfreeze_image_blocks": 1,
        "unfreeze_text_layers": 2,
        "early_stopping_start_epoch": 6,
        "early_stopping_patience": 0,
        "checkpoint_dir": "checkpoints",
        "checkpoint_name": "best.pt",
        "latest_checkpoint_name": "latest.pt",
        "log_every": 20,
        "selection_metric": "vqa_score",
        "use_scheduler": True,
        "scheduler": "plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "warmup_ratio": 0.1,
        "min_lr": 1e-6,
        "label_smoothing": 0.0,
        "backbone_lr": 2e-5,
        "head_lr": 1e-4,
        "backbone_min_lr": 1e-6,
        "head_min_lr": 5e-6,
        "early_stopping_min_delta": 0.0,
        "trainable_vilt_layers": 12,
    },
    "runtime": {
        "run_dir": "runs",
        "run_name": None,
        "use_run_dir": False,
    },
    "tracking": {
        "wandb": {
            "enabled": False,
            "project": "multimodal-vqa",
            "entity": None,
            "tags": [],
            "log_checkpoints": False,
        }
    },
    "infer": {"topk": 5},
}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, loaded)


def apply_runtime_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    resolved = deepcopy(config)
    mapping = {
        "device": (None, "device"),
        "data_root": ("data", "root"),
        "answer_vocab_path": ("data", "answer_vocab_path"),
        "checkpoint_dir": ("train", "checkpoint_dir"),
        "epochs": ("train", "epochs"),
        "max_train_samples": ("data", "max_train_samples"),
        "max_val_samples": ("data", "max_val_samples"),
        "run_name": ("runtime", "run_name"),
        "run_dir": ("runtime", "run_dir"),
        "wandb_enabled": ("tracking", "wandb", "enabled"),
        "wandb_project": ("tracking", "wandb", "project"),
        "wandb_tags": ("tracking", "wandb", "tags"),
    }
    for name, value in overrides.items():
        if value is None or name not in mapping:
            continue
        path = mapping[name]
        if path[0] is None:
            resolved[path[1]] = value
            continue
        target = resolved
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    if overrides.get("run_name") is not None or overrides.get("run_dir") is not None:
        resolved["runtime"]["use_run_dir"] = True
    return resolved


def resolve_checkpoint_config(runtime_config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Use checkpoint-owned architecture and preprocessing with local runtime paths."""
    stored_config = checkpoint.get("config")
    if not isinstance(stored_config, dict):
        return deepcopy(runtime_config)

    resolved = deepcopy(runtime_config)
    stored_model = stored_config.get("model")
    if isinstance(stored_model, dict):
        resolved["model"] = deep_update(DEFAULT_CONFIG["model"], stored_model)

    stored_data = stored_config.get("data")
    if isinstance(stored_data, dict):
        for key in ("answer_vocab_size", "image_size", "max_question_length", "processor_name"):
            if key in stored_data:
                resolved["data"][key] = stored_data[key]
    return resolved


def resolve_device(device: str, allow_fallback: bool = False):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        if allow_fallback:
            print("CUDA is configured but not available; using CPU instead.")
            return torch.device("cpu")
        raise RuntimeError("CUDA is configured but torch.cuda.is_available() is False.")
    return torch.device(device)
