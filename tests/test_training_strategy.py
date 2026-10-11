from __future__ import annotations

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from vqa_project.config import load_config
from vqa_project.model import build_model
from vqa_project.training import (
    WarmupPlateauScheduler,
    accumulation_settings_for_global_batch,
    build_optimizer,
    build_scheduler,
    configure_finetune_stage,
    ddp_static_graph_enabled,
    resume_signature,
    stage_for_epoch,
    validate_resume_config,
)


class _RepeatedCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = checkpoint(self.shared, inputs, use_reentrant=True)
        return checkpoint(self.shared, inputs, use_reentrant=True)


def _static_graph_worker(rank: int, init_file: str) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        model = DistributedDataParallel(
            _RepeatedCheckpoint(),
            find_unused_parameters=True,
            static_graph=True,
        )
        inputs = torch.randn(2, 4, requires_grad=True)
        model(inputs).square().mean().backward()
        gradient = model.module.shared.weight.grad
        assert gradient is not None
        minimum = gradient.clone()
        maximum = gradient.clone()
        distributed.all_reduce(minimum, op=distributed.ReduceOp.MIN)
        distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX)
        torch.testing.assert_close(minimum, maximum)
    finally:
        distributed.destroy_process_group()


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


def test_accumulation_settings_preserve_global_batch_for_ddp_config_semantics() -> None:
    settings = accumulation_settings_for_global_batch(128, batch_size_per_gpu=2, world_size=2)

    assert settings == {
        "config_gradient_accumulation_steps": 64,
        "runtime_gradient_accumulation_steps_per_rank": 32,
    }


def test_beit3_static_graph_is_limited_to_fixed_full_finetuning_with_checkpointing() -> None:
    config = load_config("configs/kaggle_beit3_a_t4x2.yaml")
    model_cfg = config["model"]
    train_cfg = config["train"]

    assert ddp_static_graph_enabled(model_cfg, train_cfg)
    assert not ddp_static_graph_enabled({**model_cfg, "gradient_checkpointing": False}, train_cfg)
    assert not ddp_static_graph_enabled({**model_cfg, "name": "vilt"}, train_cfg)
    assert not ddp_static_graph_enabled(model_cfg, {**train_cfg, "staged_finetuning": True})


@pytest.mark.skipif(not distributed.is_gloo_available(), reason="Gloo is not available")
def test_ddp_static_graph_supports_reentrant_checkpointing(tmp_path) -> None:
    multiprocessing.spawn(_static_graph_worker, args=(str(tmp_path / "ddp-init"),), nprocs=2, join=True)


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


def test_beit3_resume_signature_ignores_machine_asset_paths() -> None:
    original = load_config("configs/kaggle_beit3_b_vg_t4x2.yaml")
    changed = load_config("configs/kaggle_beit3_b_vg_t4x2.yaml")
    changed["model"]["cache_dir"] = "/different/cache"
    changed["data"]["visual_genome_qa_path"] = "/different/run/visual_genome_qa_train.json"

    assert resume_signature(original) == resume_signature(changed)
    validate_resume_config(changed, {"format_version": 4, "config": original})

    no_visual_genome = load_config("configs/kaggle_beit3_a_t4x2.yaml")
    assert resume_signature(original) != resume_signature(no_visual_genome)


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
