from __future__ import annotations

import json

from scripts.prepare_visual_genome_qa import prepare_manifest
from vqa_project.answers import AnswerVocab


def test_visual_genome_filter_excludes_validation_unknown_and_oov_pairs(tmp_path) -> None:
    root = tmp_path / "vqa"
    root.mkdir()
    (root / "v2_OpenEnded_mscoco_train2014_questions.json").write_text(
        json.dumps({"questions": [{"image_id": 100, "question_id": 1, "question": "train"}]}),
        encoding="utf-8",
    )
    (root / "v2_OpenEnded_mscoco_val2014_questions.json").write_text(
        json.dumps({"questions": [{"image_id": 200, "question_id": 2, "question": "val"}]}),
        encoding="utf-8",
    )
    vocab = AnswerVocab(["yes", *(f"other-{index}" for index in range(4999))], normalization="vqa_v2")
    vocab_path = tmp_path / "answer_vocab.json"
    vocab.save(vocab_path)
    image_data = tmp_path / "image_data.json"
    image_data.write_text(
        json.dumps(
            [
                {"image_id": 10, "coco_id": 100},
                {"image_id": 20, "coco_id": 200},
                {"image_id": 30, "coco_id": 300},
                {"image_id": 40, "coco_id": None},
            ]
        ),
        encoding="utf-8",
    )
    qa_path = tmp_path / "question_answers.json"
    qa_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 10,
                    "qas": [
                        {"qa_id": 1, "question": "Is it blue?", "answer": "yes"},
                        {"qa_id": 2, "question": "Is it blue?", "answer": "yes"},
                        {"qa_id": 3, "question": "What is it?", "answer": "not-in-vocab"},
                    ],
                },
                {"image_id": 20, "qas": [{"qa_id": 4, "question": "val?", "answer": "yes"}]},
                {"image_id": 30, "qas": [{"qa_id": 5, "question": "unknown?", "answer": "yes"}]},
                {"image_id": 40, "qas": [{"qa_id": 6, "question": "unmapped?", "answer": "yes"}]},
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "filtered.json"
    manifest = prepare_manifest(root, image_data, qa_path, vocab_path, output_path)

    assert [item["qa_id"] for item in manifest["qas"]] == [1]
    assert manifest["counts"]["excluded_validation_image"] == 1
    assert manifest["counts"]["excluded_nontraining_coco_image"] == 1
    assert manifest["counts"]["excluded_missing_coco_mapping"] == 1
    assert manifest["counts"]["excluded_duplicate"] == 1
    assert manifest["counts"]["excluded_answer_outside_vocab"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["qas"] == manifest["qas"]
