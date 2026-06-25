from __future__ import annotations

import csv
import json

from scripts.summarize_experiments import summarize_run, write_release_decision, write_summary


def test_experiment_summary_selects_best_vqa_epoch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "epoch": 1,
            "stage": "frozen",
            "val_vqa_score": 0.4,
            "val_accuracy": 0.3,
            "val_top5_vqa_score": 0.7,
            "trainable_parameters": 10,
            "total_seconds": 100,
        },
        {
            "epoch": 2,
            "stage": "partial",
            "val_vqa_score": 0.5,
            "val_accuracy": 0.35,
            "val_top5_vqa_score": 0.8,
            "trainable_parameters": 20,
            "total_seconds": 200,
        },
    ]
    with (run_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "run_metadata.json").write_text(json.dumps({"device": "cuda", "git_commit": "abc"}), encoding="utf-8")
    (run_dir / "official_vqa_metrics.json").write_text(json.dumps({"overall": 51.2}), encoding="utf-8")

    summary = summarize_run("finetune", run_dir)
    assert summary["best_epoch"] == 2
    assert summary["official_vqa_score"] == 51.2
    outputs = write_summary([summary], tmp_path / "summary")
    assert all(path.is_file() for path in outputs.values())


def test_release_gate_requires_one_official_vqa_point(tmp_path) -> None:
    rows = [
        {"run": "frozen", "official_vqa_score": 50.0},
        {"run": "finetune", "official_vqa_score": 51.2},
    ]
    output = tmp_path / "summary"
    output.mkdir()
    decision = write_release_decision(rows, "frozen", "finetune", 1.0, output)
    assert decision["status"] == "promote"
    assert decision["promote_candidate"] is True
    assert (output / "release_decision.json").is_file()


def test_release_gate_is_blocked_without_official_scores(tmp_path) -> None:
    rows = [
        {"run": "frozen", "official_vqa_score": None},
        {"run": "finetune", "official_vqa_score": 51.2},
    ]
    output = tmp_path / "summary"
    output.mkdir()
    decision = write_release_decision(rows, "frozen", "finetune", 1.0, output)

    assert decision["status"] == "blocked"
    assert decision["promote_candidate"] is False
    assert "Official VQA scores" in decision["reason"]
