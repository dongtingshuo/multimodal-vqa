from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.run_error_analysis import main as error_analysis_main
from scripts.run_model_comparison import run as run_model_comparison


def test_run_model_comparison_script_generates_bilingual_reports(tmp_path: Path) -> None:
    config_path = tmp_path / "demo_comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "device": "cpu",
                "output_dir": str(tmp_path / "comparison"),
                "data": {
                    "answer_vocab_size": 5,
                    "image_size": 32,
                    "max_question_length": 6,
                    "num_samples": 4,
                },
                "model": {
                    "text_model_name": "distilbert-base-uncased",
                    "hidden_dim": 16,
                    "num_attention_heads": 4,
                    "dropout": 0.0,
                    "freeze_backbones": True,
                    "pretrained_cnn": False,
                    "mock_backbones": True,
                    "mock_hidden_size": 16,
                },
                "variants": ["text_only", "image_only", "baseline_concat", "cross_attention"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    paths = run_model_comparison(str(config_path))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")
    assert len(payload["results"]) == 4
    assert paths["csv"].exists()
    assert "Experiment Objective / 实验目的" in report
    assert "Current Limitations / 当前限制" in report
    assert "not a real benchmark" in report


def test_run_error_analysis_script_prints_output_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"question":"What color?","ground_truth":"blue","prediction":"blue"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_error_analysis.py",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
        ],
    )
    error_analysis_main()
    captured = capsys.readouterr()
    assert "wrote json:" in captured.out
    assert (output_dir / "error_analysis.json").exists()
    assert (output_dir / "error_analysis.md").exists()
