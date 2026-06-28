from __future__ import annotations

import json
from pathlib import Path

from kaggle_finetune_kernel import run_kaggle_finetune


def test_missing_coco_images_use_the_official_s3_bucket(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"image_id": 42}]}), encoding="utf-8")
    requested_urls = []

    def fake_download(url, output):
        requested_urls.append(url)
        Path(output).write_bytes(b"jpeg")

    monkeypatch.setattr(run_kaggle_finetune.urllib.request, "urlretrieve", fake_download)
    run_kaggle_finetune.prepare_val_images(source, target, questions)

    assert requested_urls == [
        "https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg"
    ]
    assert (target / "COCO_val2014_000000000042.jpg").is_file()
