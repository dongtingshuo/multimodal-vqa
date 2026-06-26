from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn


def stage_for_epoch(model_cfg: dict[str, Any], train_cfg: dict[str, Any], epoch: int) -> str:
    if not train_cfg.get("staged_finetuning", False):
        return "frozen" if model_cfg.get("freeze_backbones", True) else "full"
    return "frozen" if epoch <= int(train_cfg.get("freeze_epochs", 2)) else "partial"


def configure_finetune_stage(model: nn.Module, stage: str, train_cfg: dict[str, Any]) -> None:
    configure = getattr(model, "set_finetune_stage", None)
    if callable(configure):
        configure(
            stage,
            image_blocks=int(train_cfg.get("unfreeze_image_blocks", 1)),
            text_layers=int(train_cfg.get("unfreeze_text_layers", 2)),
        )
        return
    if stage == "frozen" and hasattr(model, "freeze_backbones"):
        model.freeze_backbones()
        return
    if stage != "frozen":
        raise ValueError(f"Model {model.__class__.__name__} does not support fine-tune stage '{stage}'.")


def build_optimizer(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.AdamW:
    learning_rates = {
        "head": float(train_cfg["lr"]),
        "image": float(train_cfg.get("image_lr", train_cfg["lr"])),
        "text": float(train_cfg.get("text_lr", train_cfg["lr"])),
    }
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        family = (
            "image" if name.startswith("image_encoder.") else "text" if name.startswith("text_encoder.") else "head"
        )
        use_decay = not (name.endswith(".bias") or "norm" in name.lower())
        grouped.setdefault((family, use_decay), []).append(parameter)

    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    parameter_groups = []
    for (family, use_decay), parameters in grouped.items():
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rates[family],
                "weight_decay": weight_decay if use_decay else 0.0,
                "group_name": f"{family}_{'decay' if use_decay else 'no_decay'}",
            }
        )
    return torch.optim.AdamW(parameter_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if not train_cfg.get("use_scheduler", True):
        return None
    scheduler_name = str(train_cfg.get("scheduler", "plateau")).lower()
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 1)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )
    if scheduler_name == "warmup_cosine":
        warmup_epochs = max(0, int(round(total_epochs * float(train_cfg.get("warmup_ratio", 0.1)))))
        total_epochs = max(int(total_epochs), 1)

        def lr_lambda(epoch_index: int) -> float:
            epoch_number = epoch_index + 1
            if warmup_epochs > 0 and epoch_number <= warmup_epochs:
                return max(epoch_number / warmup_epochs, 1e-8)
            decay_epochs = max(total_epochs - warmup_epochs, 1)
            progress = min(max((epoch_number - warmup_epochs) / decay_epochs, 0.0), 1.0)
            return max(0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item(), 1e-8)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError("Unsupported train.scheduler. Expected 'plateau' or 'warmup_cosine'.")


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    metric: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def resume_signature(config: dict[str, Any]) -> str:
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    payload = {
        "model": model_cfg,
        "data": {
            key: data_cfg.get(key)
            for key in (
                "train_split",
                "val_split",
                "answer_vocab_size",
                "max_train_samples",
                "max_val_samples",
                "image_size",
                "max_question_length",
            )
        },
        "train": {
            key: train_cfg.get(key)
            for key in (
                "batch_size",
                "lr",
                "image_lr",
                "text_lr",
                "weight_decay",
                "grad_clip_norm",
                "gradient_accumulation_steps",
                "staged_finetuning",
                "freeze_epochs",
                "unfreeze_image_blocks",
                "unfreeze_text_layers",
                "selection_metric",
                "use_scheduler",
                "scheduler",
                "scheduler_factor",
                "scheduler_patience",
                "warmup_ratio",
                "min_lr",
                "label_smoothing",
                "early_stopping_start_epoch",
                "early_stopping_patience",
            )
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_resume_config(config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    if int(checkpoint.get("format_version", 0)) < 3:
        raise ValueError("Only checkpoint format v3 or newer can resume training; older files remain inference-only.")
    stored_config = checkpoint.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError("Resume checkpoint does not contain a complete training config.")
    if resume_signature(config) != resume_signature(stored_config):
        raise ValueError(
            "Resume config is incompatible with the checkpoint. Model, preprocessing, optimizer, "
            "gradient accumulation, and fine-tune schedule must match."
        )
