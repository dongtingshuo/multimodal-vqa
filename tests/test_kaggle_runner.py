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


def test_dependency_install_reuses_usable_preinstalled_torch(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(run_kaggle_finetune, "preinstalled_torch_is_usable", lambda: True)
    monkeypatch.setattr(run_kaggle_finetune, "run", lambda command, cwd=None: commands.append(command))

    run_kaggle_finetune.install_training_dependencies()

    assert len(commands) == 1
    assert run_kaggle_finetune.TRANSFORMERS_SPEC in commands[0]
    assert "wandb>=0.17" not in commands[0]
    assert not any(str(part).startswith("torch==") for part in commands[0])


def test_dependency_install_falls_back_to_pinned_torch(tmp_path: Path, monkeypatch) -> None:
    commands = []
    runtime_dir = tmp_path / "pytorch-runtime"
    monkeypatch.setattr(run_kaggle_finetune, "PYTORCH_RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(run_kaggle_finetune, "preinstalled_torch_is_usable", lambda: False)
    monkeypatch.setattr(run_kaggle_finetune, "torch_runtime_is_usable", lambda: True)
    monkeypatch.setattr(run_kaggle_finetune, "run", lambda command, cwd=None: commands.append(command))

    run_kaggle_finetune.install_training_dependencies()

    assert len(commands) == 2
    assert "--target" in commands[0]
    assert runtime_dir in commands[0]
    assert f"torch=={run_kaggle_finetune.TORCH_VERSION}" in commands[0]
    assert f"torchvision=={run_kaggle_finetune.TORCHVISION_VERSION}" in commands[0]
    assert run_kaggle_finetune.TRANSFORMERS_SPEC in commands[1]
    assert "wandb>=0.17" not in commands[1]


def test_kaggle_runner_has_no_wandb_secret_path() -> None:
    source = Path(run_kaggle_finetune.__file__).read_text(encoding="utf-8")

    assert "kaggle_secrets" not in source
    assert "WANDB_API_KEY" not in source
    assert '"--no-wandb"' in source
