from __future__ import annotations

import pytest

from vqa_project.vqa_normalization import (
    normalize_vqa_answer,
    normalize_vqa_answer_group,
    official_vqa_metrics,
    vqa_soft_score,
    vqa_soft_targets,
)


def test_vqa_normalization_handles_articles_numbers_and_punctuation() -> None:
    assert normalize_vqa_answer("The two cats.") == "2 cats"
    assert normalize_vqa_answer_group(["two", "2"]) == ["2", "2"]


def test_vqa_answer_group_preserves_unanimous_raw_text() -> None:
    assert normalize_vqa_answer_group(["Yes"] * 10) == ["Yes"] * 10
    assert vqa_soft_score(["Yes"] * 10, "yes") == 0.0


def test_official_leave_one_annotator_score_and_soft_targets() -> None:
    answers = ["yes"] * 3 + ["no"] * 7
    assert vqa_soft_score(answers, "yes") == pytest.approx(0.9)
    assert vqa_soft_targets(answers) == {"yes": pytest.approx(0.9), "no": pytest.approx(1.0)}


def test_official_metrics_require_exact_prediction_ids_and_report_answer_types() -> None:
    ground_truth = [
        {"question_id": 1, "answers": ["yes"] * 10, "answer_type": "yes/no", "question_type": "Is it?"},
        {"question_id": 2, "answers": ["no"] * 10, "answer_type": "yes/no", "question_type": "Is it?"},
    ]
    metrics = official_vqa_metrics(
        [{"question_id": 1, "answer": "yes"}, {"question_id": 2, "answer": "yes"}],
        ground_truth,
    )
    assert metrics["overall"] == 50.0
    assert metrics["per_answer_type"] == {"yes/no": 50.0}
    with pytest.raises(ValueError, match="do not match"):
        official_vqa_metrics([{ "question_id": 1, "answer": "yes" }], ground_truth)
    with pytest.raises(ValueError, match="duplicate question IDs"):
        official_vqa_metrics(
            [{"question_id": 1, "answer": "yes"}, {"question_id": 1, "answer": "yes"}],
            ground_truth,
        )
