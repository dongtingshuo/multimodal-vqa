from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

QUESTION_TYPE_KEYWORDS = {
    "color": ("color", "colour", "red", "blue", "green", "yellow", "black", "white", "颜色"),
    "count": ("how many", "number of", "count", "many", "多少", "几个", "几只", "几辆"),
    "location": ("where", "left", "right", "front", "behind", "top", "bottom", "位置", "哪里", "在哪"),
    "object": ("what", "which", "object", "thing", "animal", "person", "什么", "哪", "物体"),
    "yes_no": ("yes", "no", "是否", "是不是"),
}
YES_NO_PREFIXES = ("is ", "are ", "does ", "do ", "can ", "has ", "have ")


def classify_question_type(question: str | None) -> str:
    stripped = (question or "").strip().lower()
    normalized = f" {stripped} "
    for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return question_type
    if stripped.startswith(YES_NO_PREFIXES):
        return "yes_no"
    return "other"


def _normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]).strip().lower() if value else ""
    if value is None:
        return ""
    return str(value).strip().lower()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = {"question": "", "ground_truth": "", "prediction": "", "parse_error": True}
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _build_markdown(payload: dict[str, Any]) -> str:
    type_rows = "\n".join(
        f"| {question_type} | {stats['total']} | {stats['correct']} | {stats['accuracy']:.4f} |"
        for question_type, stats in payload["question_type_statistics"].items()
    )
    if not type_rows:
        type_rows = "| none | 0 | 0 | 0.0000 |"

    wrong_rows = "\n".join(
        f"| {item['prediction']} | {item['count']} |" for item in payload["common_wrong_predictions"]
    )
    if not wrong_rows:
        wrong_rows = "| none | 0 |"

    return f"""# Error Analysis / 错误分析

## Error Analysis Overview / 错误分析概述

Total examples: {payload['total']}

Correct examples: {payload['correct']}

Accuracy: {payload['accuracy']:.4f}

样本总数：{payload['total']}

正确样本数：{payload['correct']}

准确率：{payload['accuracy']:.4f}

## Question Type Statistics / 问题类型统计

| Question Type | Total | Correct | Accuracy |
|---|---:|---:|---:|
{type_rows}

## Common Wrong Predictions / 常见错误预测

| Prediction | Count |
|---|---:|
{wrong_rows}

## Analysis Notes / 分析说明

Question types are assigned by simple keyword rules. A prediction is counted as correct when the normalized prediction exactly matches the normalized ground-truth answer.

问题类型由简单关键词规则划分。预测答案与标准答案在归一化后完全一致时计为正确。

## Current Limitations / 当前限制

This analysis is a lightweight diagnostic view. It does not replace official VQA scoring, human review, or a full evaluation protocol.

该分析是轻量诊断视图，不能替代官方 VQA 评分、人工复核或完整评估流程。
"""


def analyze_errors(prediction_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    prediction_file = Path(prediction_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(prediction_file)
    type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    wrong_predictions: Counter[str] = Counter()
    correct = 0

    for row in rows:
        question = row.get("question", "")
        question_type = classify_question_type(question)
        ground_truth = _normalize_answer(row.get("ground_truth"))
        prediction = _normalize_answer(row.get("prediction"))
        is_correct = bool(prediction) and prediction == ground_truth
        type_stats[question_type]["total"] += 1
        if is_correct:
            correct += 1
            type_stats[question_type]["correct"] += 1
        else:
            wrong_predictions[prediction or "<empty>"] += 1

    question_type_statistics = {}
    for question_type in ["color", "count", "object", "location", "yes_no", "other"]:
        stats = type_stats.get(question_type, {"total": 0, "correct": 0})
        total = stats["total"]
        question_type_statistics[question_type] = {
            "total": total,
            "correct": stats["correct"],
            "accuracy": stats["correct"] / total if total else 0.0,
        }

    total_examples = len(rows)
    payload = {
        "prediction_path": str(prediction_file),
        "total": total_examples,
        "correct": correct,
        "accuracy": correct / total_examples if total_examples else 0.0,
        "question_type_statistics": question_type_statistics,
        "common_wrong_predictions": [
            {"prediction": prediction, "count": count}
            for prediction, count in wrong_predictions.most_common(10)
        ],
    }

    json_path = output_path / "error_analysis.json"
    markdown_path = output_path / "error_analysis.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_build_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
