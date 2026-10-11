from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa_project.answers import AnswerVocab, build_answer_vocab
from vqa_project.data import default_annotation_path, default_question_path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "images", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Expected a JSON array of Visual Genome records.")


def _question_key(text: str) -> str:
    return " ".join(text.casefold().split())


def prepare_manifest(
    vqa_root: str | Path,
    image_data_path: str | Path,
    question_answers_path: str | Path,
    answer_vocab_path: str | Path,
    output_path: str | Path,
) -> dict:
    root = Path(vqa_root)
    image_data_path = Path(image_data_path)
    question_answers_path = Path(question_answers_path)
    vocab_path = Path(answer_vocab_path)
    if not vocab_path.is_file():
        vocab = build_answer_vocab(
            default_annotation_path(root, "train"),
            top_k=5000,
            normalization="vqa_v2",
        )
        vocab.save(vocab_path)
    vocab = AnswerVocab.load(vocab_path)
    if vocab.normalization != "vqa_v2" or len(vocab) != 5000:
        raise ValueError("Visual Genome filtering requires the VQA v2-normalized Top-5000 training vocabulary.")

    train_questions = _read_json(default_question_path(root, "train"))["questions"]
    val_questions = _read_json(default_question_path(root, "val"))["questions"]
    train_image_ids = {int(item["image_id"]) for item in train_questions}
    val_image_ids = {int(item["image_id"]) for item in val_questions}
    if train_image_ids & val_image_ids:
        raise ValueError("VQA v2 train and validation image IDs unexpectedly overlap.")

    image_rows = _rows(_read_json(image_data_path))
    coco_by_vg_id = {}
    for item in image_rows:
        try:
            vg_id = int(item["image_id"])
            coco_id = int(item["coco_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if coco_id >= 0:
            coco_by_vg_id[vg_id] = coco_id

    qa_rows = _rows(_read_json(question_answers_path))
    counts: Counter[str] = Counter()
    output_rows = []
    seen = set()
    for image_row in qa_rows:
        image_id = image_row.get("image_id", image_row.get("id"))
        try:
            vg_image_id = int(image_id)
        except (TypeError, ValueError):
            counts["excluded_invalid_image_id"] += len(image_row.get("qas", []))
            continue
        qas = image_row.get("qas", image_row.get("qa", [])) or []
        counts["input_qas"] += len(qas)
        coco_id = coco_by_vg_id.get(vg_image_id)
        if coco_id is None:
            counts["excluded_missing_coco_mapping"] += len(qas)
            continue
        if coco_id in val_image_ids:
            counts["excluded_validation_image"] += len(qas)
            continue
        if coco_id not in train_image_ids:
            counts["excluded_nontraining_coco_image"] += len(qas)
            continue

        for qa in qas:
            question = " ".join(str(qa.get("question", "")).split())
            answer = " ".join(str(qa.get("answer", "")).split())
            if not question or not answer:
                counts["excluded_empty_text"] += 1
                continue
            try:
                qa_id = int(qa["qa_id"])
            except (KeyError, TypeError, ValueError):
                counts["excluded_invalid_qa_id"] += 1
                continue
            answer_index = vocab.encode(answer)
            if answer_index is None:
                counts["excluded_answer_outside_vocab"] += 1
                continue
            normalized_answer = vocab.decode(answer_index)
            duplicate_key = (coco_id, _question_key(question), normalized_answer)
            if duplicate_key in seen:
                counts["excluded_duplicate"] += 1
                continue
            seen.add(duplicate_key)
            output_rows.append(
                {
                    "qa_id": qa_id,
                    "image_id": coco_id,
                    "question": question,
                    "answer": answer,
                    "answer_vocab_index": answer_index,
                    "vg_image_id": vg_image_id,
                }
            )
            counts["retained"] += 1

    if not output_rows:
        raise ValueError("No Visual Genome QA pairs passed the COCO train-image and vocabulary filters.")
    output = {
        "schema_version": 1,
        "source": "Visual Genome v1.2 question_answers.json + image_data.json",
        "split": "COCO train2014 only",
        "answer_normalization": "vqa_v2",
        "answer_vocab_size": len(vocab),
        "counts": dict(sorted(counts.items())),
        "source_sha256": {
            "image_data": _sha256(image_data_path),
            "question_answers": _sha256(question_answers_path),
            "answer_vocab": _sha256(vocab_path),
        },
        "qas": output_rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Visual Genome QA pairs to leakage-safe COCO train images.")
    parser.add_argument("--vqa-root", required=True)
    parser.add_argument("--image-data", required=True)
    parser.add_argument("--question-answers", required=True)
    parser.add_argument("--answer-vocab", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_manifest(
        args.vqa_root,
        args.image_data,
        args.question_answers,
        args.answer_vocab,
        args.output,
    )
    print(json.dumps({"counts": result["counts"], "source_sha256": result["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
