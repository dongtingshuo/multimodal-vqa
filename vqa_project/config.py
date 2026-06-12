from __future__ import annotations

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
        "num_workers": 0,
    },
    "model": {
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
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "checkpoint_dir": "checkpoints",
        "checkpoint_name": "best.pt",
        "log_every": 20,
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


def resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
