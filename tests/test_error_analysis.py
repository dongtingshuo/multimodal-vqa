from __future__ import annotations

import json
from pathlib import Path

from vqa_project.analysis import analyze_errors, classify_question_type


def test_classify_question_type_uses_expected_keywords() -> None:
    assert classify_question_type("What color is the bus?") == "color"
    assert classify_question_type("How many chairs are there?") == "count"
    assert classify_question_type("What is on the table?") == "object"
    assert classify_question_type("Where is the person?") == "location"
    assert classify_question_type("Is the light on?") == "yes_no"
    assert classify_question_type("Describe the scene") == "other"


def test_analyze_errors_handles_empty_file(tmp_path: Path) -> None:
    predictions = tmp_path / "empty.jsonl"
    predictions.write_text("", encoding="utf-8")
    paths = analyze_errors(predictions, tmp_path / "analysis")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    report = paths["markdown"].read_text(encoding="utf-8")
    assert payload["total"] == 0
    assert payload["accuracy"] == 0.0
    assert "Error Analysis Overview / 错误分析概述" in report


def test_analyze_errors_generates_statistics_and_bilingual_report(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                '{"question":"What color is it?","ground_truth":"red","prediction":"red"}',
                '{"question":"How many are there?","ground_truth":"2","prediction":"3"}',
                '{"question":"Is it open?","ground_truth":"yes","prediction":"no"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = analyze_errors(predictions, tmp_path / "analysis")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    report = paths["markdown"].read_text(encoding="utf-8")
    assert payload["total"] == 3
    assert payload["correct"] == 1
    assert payload["question_type_statistics"]["color"]["accuracy"] == 1.0
    assert payload["common_wrong_predictions"][0]["prediction"] in {"3", "no"}
    assert "Question Type Statistics / 问题类型统计" in report
    assert "Current Limitations / 当前限制" in report
