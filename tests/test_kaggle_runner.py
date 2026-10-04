from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_finetune_kernel import run_kaggle_finetune


def test_missing_coco_images_use_the_official_s3_bucket(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"image_id": 42}]}), encoding="utf-8")
    requested_urls = []

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_download(request, timeout):
        requested_urls.append(request.full_url)
        assert timeout == 60
        return FakeResponse(b"jpeg")

    monkeypatch.setattr(run_kaggle_finetune.urllib.request, "urlopen", fake_download)
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


def test_kaggle_runner_streams_child_output_to_console_and_run_log(tmp_path: Path, monkeypatch, capsys) -> None:
    log_path = tmp_path / "runner.log"
    monkeypatch.setattr(run_kaggle_finetune, "RUN_LOG_PATH", log_path)

    run_kaggle_finetune.run(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('child stdout'); print('child stderr', file=sys.stderr); sys.stdout.write('step=1\\r'); sys.stdout.flush()",
        ],
        cwd=tmp_path,
    )

    output = capsys.readouterr().out
    assert "child stdout" in output
    assert "child stderr" in output
    assert "step=1" in output
    assert "child stdout" in log_path.read_text(encoding="utf-8")
    assert "step=1" in log_path.read_text(encoding="utf-8")


def test_progress_stage_records_current_state(tmp_path: Path, monkeypatch, capsys) -> None:
    log_path = tmp_path / "runner.log"
    status_path = tmp_path / "runner_status.json"
    monkeypatch.setattr(run_kaggle_finetune, "RUN_LOG_PATH", log_path)
    monkeypatch.setattr(run_kaggle_finetune, "RUN_STATUS_PATH", status_path)
    monkeypatch.setattr(run_kaggle_finetune, "RUN_STARTED_AT", run_kaggle_finetune.time.monotonic())

    with run_kaggle_finetune.progress_stage("test-stage"):
        pass

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "running"
    assert status["stage"] == "test-stage"
    assert "stage completed" in status["message"]
    assert "test-stage" in log_path.read_text(encoding="utf-8")


def test_t4x2_runner_fails_fast_when_only_one_gpu_is_visible(monkeypatch) -> None:
    monkeypatch.setattr(run_kaggle_finetune, "VQA_NUM_GPUS", "2")
    monkeypatch.setattr(
        run_kaggle_finetune.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="1\n"),
    )

    with pytest.raises(ValueError, match="exposes only 1 CUDA GPU"):
        run_kaggle_finetune.training_launcher()


def test_runner_can_explicitly_use_one_gpu(monkeypatch) -> None:
    monkeypatch.setattr(run_kaggle_finetune, "VQA_NUM_GPUS", "1")
    monkeypatch.setattr(
        run_kaggle_finetune.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="2\n"),
    )

    assert run_kaggle_finetune.training_launcher() == ["python", "-u", "train.py"]


def test_kaggle_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(zipfile.BadZipFile, match="Unsafe archive member"):
        run_kaggle_finetune.extract_zip_safely(archive_path, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()


def test_kaggle_download_rejects_non_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Download URL"):
        run_kaggle_finetune.download_https("file:///tmp/data.zip", tmp_path / "data.zip")
