from __future__ import annotations

import csv

from vqa_project.tracking import (
    create_run_directory,
    init_wandb_tracker,
    save_training_curves,
    write_run_summary,
    write_training_history,
)


def test_training_history_and_curves_are_written(tmp_path) -> None:
    rows = [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "train_accuracy": 0.2,
            "train_vqa_score": 0.3,
            "train_top5_vqa_score": 0.6,
            "val_loss": 1.8,
            "val_accuracy": 0.25,
            "val_vqa_score": 0.35,
            "val_top5_vqa_score": 0.65,
            "learning_rate": 1e-4,
            "epoch_seconds": 1.0,
            "total_seconds": 1.0,
            "best_checkpoint": True,
        }
    ]
    history_path = tmp_path / "history.csv"
    curve_path = tmp_path / "curves.png"
    write_training_history(history_path, rows)
    save_training_curves(curve_path, rows)

    with history_path.open(encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert saved[0]["val_vqa_score"] == "0.35"
    assert curve_path.stat().st_size > 0


def test_run_directory_and_summary_are_written(tmp_path) -> None:
    run_dir = create_run_directory(tmp_path, "strong run")
    assert run_dir.name.endswith("-strong-run")
    summary_path = run_dir / "run_summary.json"
    write_run_summary(
        summary_path,
        [
            {
                "epoch": 1,
                "val_vqa_score": 0.4,
                "best_checkpoint": False,
            },
            {
                "epoch": 2,
                "val_vqa_score": 0.5,
                "best_checkpoint": True,
            },
        ],
        {"total_seconds": 12.0, "artifacts": {"checkpoint": "best.pt"}},
    )
    assert '"best_epoch": 2' in summary_path.read_text(encoding="utf-8")


def test_wandb_tracker_is_noop_when_disabled() -> None:
    tracker = init_wandb_tracker({"wandb": {"enabled": False}}, {"seed": 42}, run_name="test")
    tracker.log({"metric": 1.0})
    tracker.finish()
    assert tracker.enabled is False
