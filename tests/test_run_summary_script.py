from __future__ import annotations

import csv
import json

from scripts.summarize_runs import summarize_run, write_rows


def test_summarize_run_reads_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "20260626-test"
    run_dir.mkdir()
    history_path = run_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "stage",
                "val_loss",
                "val_accuracy",
                "val_vqa_score",
                "val_top5_vqa_score",
                "learning_rate",
                "total_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "stage": "partial",
                "val_loss": 1.0,
                "val_accuracy": 0.5,
                "val_vqa_score": 0.6,
                "val_top5_vqa_score": 0.8,
                "learning_rate": 1e-4,
                "total_seconds": 3600,
            }
        )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"artifacts": {"checkpoint": "best.pt"}, "git_commit": "abc"}),
        encoding="utf-8",
    )

    row = summarize_run(run_dir)
    assert row is not None
    assert row["val_vqa_score"] == 0.6
    assert row["checkpoint"] == "best.pt"

    output = tmp_path / "summary.csv"
    write_rows([row], output)
    assert output.read_text(encoding="utf-8").startswith("run_dir,best_epoch")
