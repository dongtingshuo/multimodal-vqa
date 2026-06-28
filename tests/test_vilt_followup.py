from __future__ import annotations

from scripts.select_vilt_followup import build_followup_config, select_followup
from vqa_project.config import load_config


def row(train: float, val: float, accuracy: float) -> dict[str, float]:
    return {"train_vqa_score": train, "val_vqa_score": val, "val_accuracy": accuracy}


def test_followup_replicates_a_successful_run() -> None:
    assert select_followup([row(0.70, 0.66, 0.56)])["branch"] == "replicate"


def test_followup_reduces_trainable_layers_for_overfitting() -> None:
    decision = select_followup([row(0.72, 0.60, 0.50)])
    assert decision["branch"] == "last_six_layers"
    assert decision["trainable_vilt_layers"] == 6


def test_followup_raises_lr_for_stable_underfitting() -> None:
    decision = select_followup([row(0.61, 0.59, 0.49), row(0.63, 0.60, 0.50)])
    assert decision["branch"] == "higher_backbone_lr"
    assert decision["backbone_lr"] == 3e-5


def test_followup_lowers_lr_for_instability() -> None:
    decision = select_followup(
        [row(0.55, 0.54, 0.45), row(0.62, 0.49, 0.43), row(0.64, 0.55, 0.46), row(0.66, 0.50, 0.44)]
    )
    assert decision["branch"] == "lower_backbone_lr"
    assert decision["backbone_lr"] == 1e-5


def test_followup_writes_to_a_separate_checkpoint_directory() -> None:
    decision = select_followup([row(0.61, 0.59, 0.49)])
    config = build_followup_config(load_config("configs/kaggle_vilt.yaml"), decision)
    assert config["train"]["checkpoint_dir"].endswith("/vilt-higher_backbone_lr")
