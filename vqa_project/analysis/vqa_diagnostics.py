from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..answers import AnswerVocab, normalize_answer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("results", [payload]))
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a JSON list of records or JSONL records in {path}.")
    return payload


def _group_metrics(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for key, row in sorted(rows.items(), key=lambda item: (-item[1]["questions"], item[0])):
        questions = int(row["questions"])
        occurrences = int(row["gold_answer_occurrences"])
        result[key] = {
            "questions": questions,
            "gold_answer_occurrences": occurrences,
            "answer_occurrence_coverage": row["in_vocab_occurrences"] / occurrences if occurrences else 0.0,
            "questions_with_any_in_vocab_answer": int(row["covered_questions"]),
            "question_coverage": row["covered_questions"] / questions if questions else 0.0,
            "vocabulary_constrained_oracle_training_score": row["oracle_score_sum"] / questions if questions else 0.0,
            "prediction_training_score": row["prediction_score_sum"] / questions if questions else 0.0,
            "prediction_exact_match_any_annotator": row["prediction_exact_match_sum"] / questions if questions else 0.0,
        }
    return result


def _markdown_report(payload: dict[str, Any]) -> str:
    coverage = payload["answer_vocabulary_coverage"]
    prediction = payload["prediction_error_analysis"]

    def table_rows(group: dict[str, dict[str, Any]]) -> str:
        return "\n".join(
            f"| {name} | {stats['questions']} | {stats['question_coverage']:.4f} | "
            f"{stats['prediction_training_score']:.4f} | {stats['prediction_exact_match_any_annotator']:.4f} |"
            for name, stats in group.items()
        ) or "| none | 0 | 0.0000 | 0.0000 | 0.0000 |"

    oov_rows = "\n".join(
        f"| {item['answer']} | {item['count']} |" for item in coverage["most_frequent_out_of_vocabulary_answers"]
    ) or "| none | 0 |"
    wrong_rows = "\n".join(
        f"| {item['answer']} | {item['count']} |" for item in prediction["most_common_zero_credit_predictions"]
    ) or "| none | 0 |"
    example_rows = "\n".join(
        f"| {item['question_id']} | {item['question']} | {item['prediction']} | {item['training_score']:.3f} |"
        for item in prediction["lowest_scoring_examples"]
    ) or "| none | - | - | 0.000 |"

    return f"""# VQA Answer Coverage and Error Diagnostics / 答案覆盖率与错误诊断

## Evaluation Scope / 评估范围

- Validation questions: {payload['validation_questions']}
- Prediction records: {payload['prediction_records']}
- Answer vocabulary size: {coverage['vocabulary_size']}
- Answer normalization: project `normalize_answer`; these diagnostics are not the official VQA score.

- 验证问题数：{payload['validation_questions']}
- 预测条数：{payload['prediction_records']}
- 答案词表大小：{coverage['vocabulary_size']}
- 答案归一化：项目内 `normalize_answer`；本诊断不等同于官方 VQA 分数。

## Vocabulary Coverage / 词表覆盖率

| Metric | Value |
|---|---:|
| Annotated answer occurrences in vocabulary | {coverage['in_vocabulary_answer_occurrences']} / {coverage['gold_answer_occurrences']} ({coverage['answer_occurrence_coverage']:.4f}) |
| Questions with at least one representable answer | {coverage['questions_with_any_in_vocab_answer']} / {coverage['validation_questions']} ({coverage['question_coverage']:.4f}) |
| Mean vocabulary-constrained oracle training score | {coverage['vocabulary_constrained_oracle_training_score']:.4f} |

| 指标 | 数值 |
|---|---:|
| 落入词表的标注答案数 | {coverage['in_vocabulary_answer_occurrences']} / {coverage['gold_answer_occurrences']} ({coverage['answer_occurrence_coverage']:.4f}) |
| 至少有一个可表示答案的问题 | {coverage['questions_with_any_in_vocab_answer']} / {payload['validation_questions']} ({coverage['question_coverage']:.4f}) |
| 词表约束下的理论平均训练软分 | {coverage['vocabulary_constrained_oracle_training_score']:.4f} |

The oracle score is the best per-question soft target available among in-vocabulary answers under this project's `min(annotation_count / 3, 1)` target rule. It is a diagnostic upper bound, not a model metric or official benchmark score.

理论分数按项目 `min(答案标注数 / 3, 1)` 规则，在词表允许的答案中逐题取最佳值后求平均。它是诊断上界，不是模型指标或官方 benchmark 成绩。

### Most Frequent Out-of-Vocabulary Answers / 高频词表外答案

| Normalized answer | Annotation count |
|---|---:|
{oov_rows}

## Prediction Error Analysis / 预测错误分析

| Metric | Value |
|---|---:|
| Mean project soft-target score | {prediction['mean_training_score']:.4f} |
| Exact match to any annotator | {prediction['exact_match_any_annotator']:.4f} |
| Zero-credit predictions | {prediction['zero_credit_count']} |
| Existing official validation score | {payload.get('official_vqa_score', 'not attached')} |

| 指标 | 数值 |
|---|---:|
| 项目软标签平均分 | {prediction['mean_training_score']:.4f} |
| 与任一标注答案完全匹配率 | {prediction['exact_match_any_annotator']:.4f} |
| 零分预测数 | {prediction['zero_credit_count']} |
| 已记录的官方验证分数 | {payload.get('official_vqa_score', '未附加')} |

Question-type breakdown uses the official VQA annotation fields. Prediction score uses the project's normalized soft-target rule; exact match means the prediction equals at least one normalized human answer.

问题类型分组使用官方 VQA 标注字段。预测软分按项目归一化规则计算；完全匹配表示预测答案与至少一条归一化人工答案相同。

### By Official Question Type / 按官方问题类型

| Type | Questions | Vocabulary coverage | Prediction soft score | Exact match |
|---|---:|---:|---:|---:|
{table_rows(prediction['by_question_type'])}

### By Official Answer Type / 按官方答案类型

| Type | Questions | Vocabulary coverage | Prediction soft score | Exact match |
|---|---:|---:|---:|---:|
{table_rows(prediction['by_answer_type'])}

### Most Common Zero-Credit Predictions / 高频零分预测

| Predicted answer | Count |
|---|---:|
{wrong_rows}

### Lowest-Scoring Examples / 低分样例

| Question ID | Question | Prediction | Project soft score |
|---:|---|---|---:|
{example_rows}

## Interpretation / 解读

Use the vocabulary coverage and category breakdown to identify whether the next experiment should address answer-space limits or model reasoning. Do not compare these project-normalized diagnostics directly with the official VQA toolkit result.

结合词表覆盖率与类别分组判断下一轮应优先处理答案空间限制还是模型推理问题。不要将项目归一化诊断值直接与官方 VQA toolkit 分数比较。
"""


def analyze_vqa_diagnostics(
    predictions_path: str | Path,
    questions_path: str | Path,
    annotations_path: str | Path,
    answer_vocab_path: str | Path,
    output_dir: str | Path,
    official_vqa_score: float | None = None,
    top_k: int = 20,
) -> dict[str, Path]:
    predictions_file = Path(predictions_path)
    questions_file = Path(questions_path)
    annotations_file = Path(annotations_path)
    vocab_file = Path(answer_vocab_path)
    output_path = Path(output_dir)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    predictions = _read_records(predictions_file)
    annotations_payload = json.loads(annotations_file.read_text(encoding="utf-8"))
    questions_payload = json.loads(questions_file.read_text(encoding="utf-8"))
    annotations = {int(row["question_id"]): row for row in annotations_payload["annotations"]}
    questions = {int(row["question_id"]): row for row in questions_payload["questions"]}
    if len(annotations) != len(annotations_payload["annotations"]):
        raise ValueError("Validation annotations contain duplicate question_id values.")
    if len(questions) != len(questions_payload["questions"]):
        raise ValueError("Validation questions contain duplicate question_id values.")
    prediction_map: dict[int, str] = {}
    for row in predictions:
        question_id = int(row["question_id"])
        if question_id in prediction_map:
            raise ValueError(f"Duplicate prediction for question_id={question_id}.")
        answer = row.get("answer", row.get("prediction"))
        if answer is None:
            raise ValueError(f"Prediction for question_id={question_id} has no answer field.")
        prediction_map[question_id] = normalize_answer(str(answer))

    annotation_ids = set(annotations)
    question_ids = set(questions)
    prediction_ids = set(prediction_map)
    if annotation_ids != question_ids:
        raise ValueError("Question and annotation files contain different question_id sets.")
    if prediction_ids != annotation_ids:
        missing = len(annotation_ids - prediction_ids)
        unexpected = len(prediction_ids - annotation_ids)
        raise ValueError(
            "Prediction IDs do not match validation annotations "
            f"(missing={missing}, unexpected={unexpected})."
        )

    answer_vocab = AnswerVocab.load(vocab_file)
    answer_set = set(answer_vocab.idx_to_answer)
    oov_answers: Counter[str] = Counter()
    coverage_totals = {
        "gold_answer_occurrences": 0,
        "in_vocabulary_answer_occurrences": 0,
        "questions_with_any_in_vocab_answer": 0,
        "oracle_score_sum": 0.0,
    }
    prediction_totals = {
        "training_score_sum": 0.0,
        "exact_match_sum": 0,
        "zero_credit_count": 0,
    }
    by_question_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_answer_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    zero_credit_predictions: Counter[str] = Counter()
    low_scoring_examples: list[dict[str, Any]] = []

    for question_id, annotation in annotations.items():
        answer_rows = annotation.get("answers") or []
        gold_answers = [normalize_answer(str(row.get("answer", ""))) for row in answer_rows]
        gold_answers = [answer for answer in gold_answers if answer]
        if not gold_answers and annotation.get("multiple_choice_answer"):
            gold_answers = [normalize_answer(str(annotation["multiple_choice_answer"]))]
        counts = Counter(gold_answers)
        in_vocab_count = sum(count for answer, count in counts.items() if answer in answer_set)
        has_vocab_answer = any(answer in answer_set for answer in counts)
        oracle_score = max(
            (min(count / 3.0, 1.0) for answer, count in counts.items() if answer in answer_set),
            default=0.0,
        )
        for answer, count in counts.items():
            if answer not in answer_set:
                oov_answers[answer] += count

        prediction = prediction_map[question_id]
        prediction_count = counts.get(prediction, 0)
        training_score = min(prediction_count / 3.0, 1.0)
        exact_match = prediction_count > 0
        question = questions[question_id]["question"]
        official_question_type = str(annotation.get("question_type") or "unknown")
        official_answer_type = str(annotation.get("answer_type") or "unknown")

        coverage_totals["gold_answer_occurrences"] += len(gold_answers)
        coverage_totals["in_vocabulary_answer_occurrences"] += in_vocab_count
        coverage_totals["questions_with_any_in_vocab_answer"] += int(has_vocab_answer)
        coverage_totals["oracle_score_sum"] += oracle_score
        prediction_totals["training_score_sum"] += training_score
        prediction_totals["exact_match_sum"] += int(exact_match)
        prediction_totals["zero_credit_count"] += int(prediction_count == 0)
        if prediction_count == 0:
            zero_credit_predictions[prediction or "<empty>"] += 1

        for key, group_name in ((official_question_type, "question"), (official_answer_type, "answer")):
            group = by_question_type[key] if group_name == "question" else by_answer_type[key]
            group["questions"] += 1
            group["gold_answer_occurrences"] += len(gold_answers)
            group["in_vocab_occurrences"] += in_vocab_count
            group["covered_questions"] += int(has_vocab_answer)
            group["oracle_score_sum"] += oracle_score
            group["prediction_score_sum"] += training_score
            group["prediction_exact_match_sum"] += int(exact_match)

        low_scoring_examples.append(
            {
                "question_id": question_id,
                "question": question,
                "prediction": prediction,
                "training_score": training_score,
                "annotator_count": prediction_count,
                "answer_type": official_answer_type,
                "question_type": official_question_type,
                "top_gold_answers": [
                    {"answer": answer, "count": count}
                    for answer, count in counts.most_common(5)
                ],
            }
        )

    total_questions = len(annotations)
    total_occurrences = int(coverage_totals["gold_answer_occurrences"])
    prediction_count = len(prediction_map)
    coverage = {
        "vocabulary_size": len(answer_vocab),
        "validation_questions": total_questions,
        "gold_answer_occurrences": total_occurrences,
        "in_vocabulary_answer_occurrences": int(coverage_totals["in_vocabulary_answer_occurrences"]),
        "answer_occurrence_coverage": coverage_totals["in_vocabulary_answer_occurrences"] / total_occurrences
        if total_occurrences
        else 0.0,
        "questions_with_any_in_vocab_answer": int(coverage_totals["questions_with_any_in_vocab_answer"]),
        "question_coverage": coverage_totals["questions_with_any_in_vocab_answer"] / total_questions
        if total_questions
        else 0.0,
        "vocabulary_constrained_oracle_training_score": coverage_totals["oracle_score_sum"] / total_questions
        if total_questions
        else 0.0,
        "most_frequent_out_of_vocabulary_answers": [
            {"answer": answer, "count": count} for answer, count in oov_answers.most_common(top_k)
        ],
    }
    prediction_analysis = {
        "mean_training_score": prediction_totals["training_score_sum"] / prediction_count if prediction_count else 0.0,
        "exact_match_any_annotator": prediction_totals["exact_match_sum"] / prediction_count
        if prediction_count
        else 0.0,
        "zero_credit_count": int(prediction_totals["zero_credit_count"]),
        "zero_credit_rate": prediction_totals["zero_credit_count"] / prediction_count if prediction_count else 0.0,
        "most_common_zero_credit_predictions": [
            {"answer": answer, "count": count} for answer, count in zero_credit_predictions.most_common(top_k)
        ],
        "by_question_type": _group_metrics(by_question_type),
        "by_answer_type": _group_metrics(by_answer_type),
        "lowest_scoring_examples": sorted(
            low_scoring_examples,
            key=lambda item: (item["training_score"], -item["annotator_count"], item["question_id"]),
        )[:top_k],
    }
    payload = {
        "predictions_path": str(predictions_file),
        "questions_path": str(questions_file),
        "annotations_path": str(annotations_file),
        "answer_vocab_path": str(vocab_file),
        "input_sha256": {
            "predictions": _sha256(predictions_file),
            "questions": _sha256(questions_file),
            "annotations": _sha256(annotations_file),
            "answer_vocab": _sha256(vocab_file),
        },
        "validation_questions": total_questions,
        "prediction_records": prediction_count,
        "official_vqa_score": official_vqa_score,
        "answer_vocabulary_coverage": coverage,
        "prediction_error_analysis": prediction_analysis,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "vqa_diagnostics.json"
    markdown_path = output_path / "vqa_diagnostics.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
