from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import validate_vqa_data


def write_split(root: Path, split: str, include_image: bool = True) -> None:
    coco_split = f"{split}2014"
    image_dir = root / coco_split
    image_dir.mkdir()
    if include_image:
        Image.new("RGB", (8, 8), color="white").save(image_dir / f"COCO_{coco_split}_000000000001.jpg")
    (root / f"v2_OpenEnded_mscoco_{coco_split}_questions.json").write_text(
        json.dumps({"questions": [{"question_id": 10, "image_id": 1, "question": "What?"}]}),
        encoding="utf-8",
    )
    (root / f"v2_mscoco_{coco_split}_annotations.json").write_text(
        json.dumps({"annotations": [{"question_id": 10, "image_id": 1, "answers": []}]}),
        encoding="utf-8",
    )


def test_strict_validation_checks_all_references(tmp_path: Path, monkeypatch) -> None:
    write_split(tmp_path, "val")
    monkeypatch.setitem(
        validate_vqa_data.EXPECTED_COUNTS,
        "val",
        {"questions": 1, "annotations": 1, "images": 1},
    )
    result = validate_vqa_data.validate_split(tmp_path, "val", sample_images=1, strict_full=True)
    assert result == {"questions": 1, "annotations": 1, "image_files": 1, "images_checked": 1}


def test_strict_validation_rejects_missing_image(tmp_path: Path, monkeypatch) -> None:
    write_split(tmp_path, "val", include_image=False)
    monkeypatch.setitem(
        validate_vqa_data.EXPECTED_COUNTS,
        "val",
        {"questions": 1, "annotations": 1, "images": 0},
    )
    with pytest.raises(FileNotFoundError, match="missing 1 referenced images"):
        validate_vqa_data.validate_split(tmp_path, "val", sample_images=1, strict_full=True)
