from __future__ import annotations

import pytest

from vqa_project.config import load_config
from vqa_project.model import build_model
from vqa_project.training import (
    build_optimizer,
    configure_finetune_stage,
    resume_signature,
    stage_for_epoch,
    validate_resume_config,
)


def model_config() -> dict:
    return {
        "name": "cross_attention",
        "answer_vocab_size": 5,
        "hidden_dim": 16,
        "num_attention_heads": 4,
        "dropout": 0.0,
        "freeze_backbones": True,
        "pretrained_cnn": False,
        "mock_backbones": True,
        "mock_hidden_size": 16,
    }


def train_config() -> dict:
    return {
        "lr": 1e-4,
        "image_lr": 1e-5,
        "text_lr": 5e-6,
        "weight_decay": 0.01,
        "staged_finetuning": True,
        "freeze_epochs": 2,
        "unfreeze_image_blocks": 1,
        "unfreeze_text_layers": 2,
    }


def test_stage_schedule_and_partial_unfreezing() -> None:
    model = build_model(model_config())
    cfg = train_config()
    assert stage_for_epoch(model_config(), cfg, 2) == "frozen"
    assert stage_for_epoch(model_config(), cfg, 3) == "partial"

    configure_finetune_stage(model, "frozen", cfg)
    assert not any(parameter.requires_grad for parameter in model.image_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())

    configure_finetune_stage(model, "partial", cfg)
    assert any(parameter.requires_grad for parameter in model.image_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.text_encoder.parameters())


def test_optimizer_uses_differential_learning_rates() -> None:
    model = build_model(model_config())
    optimizer = build_optimizer(model, train_config())
    rates = {group["group_name"].split("_")[0]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"head": 1e-4, "image": 1e-5, "text": 5e-6}
    assert any(group["weight_decay"] == 0.0 for group in optimizer.param_groups)


def test_resume_signature_ignores_runtime_paths_and_total_epochs() -> None:
    original = load_config("configs/kaggle_finetune.yaml")
    changed = load_config("configs/kaggle_finetune.yaml")
    changed["data"]["root"] = "/different/input"
    changed["train"]["checkpoint_dir"] = "/different/output"
    changed["train"]["epochs"] = 20
    assert resume_signature(original) == resume_signature(changed)
    validate_resume_config(changed, {"format_version": 3, "config": original})


def test_resume_signature_rejects_effective_training_changes() -> None:
    original = load_config("configs/kaggle_finetune.yaml")
    changed_batch = load_config("configs/kaggle_finetune.yaml")
    changed_batch["train"]["batch_size"] = 8
    changed_samples = load_config("configs/kaggle_finetune.yaml")
    changed_samples["data"]["max_train_samples"] = 1000

    assert resume_signature(original) != resume_signature(changed_batch)
    assert resume_signature(original) != resume_signature(changed_samples)


def test_legacy_checkpoint_cannot_resume() -> None:
    config = load_config("configs/default.yaml")
    with pytest.raises(ValueError, match="format v3"):
        validate_resume_config(config, {"format_version": 2, "config": config})
