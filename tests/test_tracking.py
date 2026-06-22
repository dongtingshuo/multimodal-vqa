from __future__ import annotations

import csv

from vqa_project.tracking import save_training_curves, write_training_history


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
