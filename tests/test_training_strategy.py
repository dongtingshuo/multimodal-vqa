from __future__ import annotations

import pytest

from vqa_project.config import load_config
from vqa_project.model import build_model
from vqa_project.training import (
    WarmupPlateauScheduler,
    build_optimizer,
    build_scheduler,
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


def test_vilt_optimizer_uses_backbone_and_head_learning_rates() -> None:
    model = build_model({**model_config(), "name": "vilt", "freeze_backbones": False})
    optimizer = build_optimizer(
        model,
        {**train_config(), "backbone_lr": 2e-5, "head_lr": 1e-4},
    )
    rates = {group["group_name"].split("_")[0]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"backbone": 2e-5, "head": 1e-4}


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


def test_warmup_cosine_scheduler_steps_without_metric() -> None:
    model = build_model(model_config())
    optimizer = build_optimizer(model, train_config())
    scheduler = build_scheduler(
        optimizer,
        {
            **train_config(),
            "use_scheduler": True,
            "scheduler": "warmup_cosine",
            "warmup_ratio": 0.5,
        },
        total_epochs=4,
    )
    assert scheduler is not None
    first_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] != pytest.approx(first_lr)


def test_warmup_plateau_scheduler_warms_per_optimizer_step_and_restores_state() -> None:
    model = build_model({**model_config(), "name": "vilt", "freeze_backbones": False})
    cfg = {
        **train_config(),
        "backbone_lr": 2e-5,
        "head_lr": 1e-4,
        "backbone_min_lr": 1e-6,
        "head_min_lr": 5e-6,
        "use_scheduler": True,
        "scheduler": "warmup_plateau",
        "warmup_ratio": 0.5,
    }
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_epochs=2, steps_per_epoch=2)
    assert isinstance(scheduler, WarmupPlateauScheduler)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler.step_update()
    assert all(after > before for after, before in zip([group["lr"] for group in optimizer.param_groups], initial_lrs))

    state = scheduler.state_dict()
    restored_optimizer = build_optimizer(model, cfg)
    restored = build_scheduler(restored_optimizer, cfg, total_epochs=2, steps_per_epoch=2)
    assert isinstance(restored, WarmupPlateauScheduler)
    restored.load_state_dict(state)
    assert restored.state_dict()["current_lrs"] == pytest.approx(state["current_lrs"])


def test_legacy_checkpoint_cannot_resume() -> None:
    config = load_config("configs/default.yaml")
    with pytest.raises(ValueError, match="format v3"):
        validate_resume_config(config, {"format_version": 2, "config": config})


def test_kaggle_vilt_config_disables_online_wandb() -> None:
    config = load_config("configs/kaggle_vilt.yaml")
    assert config["model"]["name"] == "vilt"
    assert config["tracking"]["wandb"]["enabled"] is False
    assert config["tracking"]["wandb"]["required"] is False
    assert config["tracking"]["wandb"]["log_checkpoints"] is False
