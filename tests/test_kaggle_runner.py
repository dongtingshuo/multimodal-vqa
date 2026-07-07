from __future__ import annotations

import json
import sys
import types
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
    assert "transformers>=4.40" in commands[0]
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
    assert "transformers>=4.40" in commands[1]


def test_wandb_secret_failure_disables_tracking(monkeypatch) -> None:
    class BrokenUserSecretsClient:
        def get_secret(self, label):
            raise ConnectionError(label)

    fake_kaggle_secrets = types.SimpleNamespace(UserSecretsClient=BrokenUserSecretsClient)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle_secrets)
    monkeypatch.setattr(run_kaggle_finetune.time, "sleep", lambda seconds: None)

    assert run_kaggle_finetune.load_wandb_api_key() is None
    assert not run_kaggle_finetune.configure_wandb(None)
