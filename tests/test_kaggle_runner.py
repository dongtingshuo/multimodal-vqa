from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_finetune_kernel import run_kaggle_finetune


def test_resolve_vqa_data_root_supports_nested_kaggle_dataset_mount(tmp_path: Path) -> None:
    data_root = tmp_path / "datasets" / "owner" / "coco2014vqa" / "Dataset"
    train_images = data_root / "images" / "train2014"
    val_images = data_root / "images" / "val2014"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)

    resolved_root, resolved_train, resolved_val = run_kaggle_finetune.resolve_vqa_data_root(
        [tmp_path / "missing", tmp_path / "datasets"]
    )

    assert resolved_root == tmp_path / "datasets"
    assert resolved_train == train_images
    assert resolved_val == val_images


def test_resolve_vqa_data_root_skips_existing_candidate_without_splits(tmp_path: Path) -> None:
    empty_mount = tmp_path / "datasets"
    empty_mount.mkdir()
    legacy_root = tmp_path / "coco2014vqa" / "Dataset"
    train_images = legacy_root / "images" / "train2014"
    val_images = legacy_root / "images" / "val2014"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)

    resolved_root, resolved_train, resolved_val = run_kaggle_finetune.resolve_vqa_data_root(
        [empty_mount, legacy_root]
    )

    assert resolved_root == legacy_root
    assert resolved_train == train_images
    assert resolved_val == val_images


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
    monkeypatch.setattr(run_kaggle_finetune, "CONFIG_PATH", "configs/kaggle_vilt_last6_t4x2.yaml")
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
    monkeypatch.setattr(run_kaggle_finetune, "CONFIG_PATH", "configs/kaggle_vilt_last6_t4x2.yaml")
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


def test_beit3_dependency_install_uses_torchscale_compatible_versions(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(run_kaggle_finetune, "CONFIG_PATH", "configs/kaggle_beit3_a_t4x2.yaml")
    monkeypatch.setattr(run_kaggle_finetune, "preinstalled_torch_is_usable", lambda: True)
    monkeypatch.setattr(run_kaggle_finetune, "run", lambda command, cwd=None: commands.append(command))

    run_kaggle_finetune.install_training_dependencies()

    beit3_install = commands[-1]
    assert "fairscale==0.4.0" in beit3_install
    assert "timm==0.6.13" in beit3_install
    assert "einops>=0.6" in beit3_install
    assert not any(str(part) == "fairscale>=0.4.13" for part in beit3_install)


def test_kaggle_runner_defaults_to_beit3_a() -> None:
    assert run_kaggle_finetune.CONFIG_PATH == "configs/kaggle_beit3_a_t4x2.yaml"
    assert run_kaggle_finetune.DEFAULT_RUN_NAME == "beit3-vqa-a-seed42-t4x2"


def test_transformers_version_is_pinned_consistently_for_beit3_tokenizer() -> None:
    expected = "transformers==4.57.6"
    project_root = Path(__file__).resolve().parents[1]

    assert run_kaggle_finetune.TRANSFORMERS_SPEC == expected
    for requirements_path in (
        project_root / "requirements.txt",
        project_root / "autodl" / "requirements.txt",
    ):
        assert expected in requirements_path.read_text(encoding="utf-8")
    assert expected in (project_root / "pyproject.toml").read_text(encoding="utf-8")


def test_runner_adds_checked_out_repository_to_import_path(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    monkeypatch.setattr(run_kaggle_finetune, "REPO_ROOT", repo_root)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(repo_root)])

    run_kaggle_finetune.ensure_repo_import_path()

    assert sys.path[0] == str(repo_root.resolve())


def test_beit3_preflight_benchmarks_generates_config_and_smokes_then_reuses(tmp_path: Path, monkeypatch) -> None:
    from scripts.prepare_beit3_runtime_config import prepare_runtime_config

    repo_root = Path(run_kaggle_finetune.__file__).resolve().parents[1]
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(run_kaggle_finetune, "REPO_ROOT", repo_root)
    monkeypatch.setattr(run_kaggle_finetune, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(run_kaggle_finetune, "CONFIG_PATH", "configs/kaggle_beit3_a_t4x2.yaml")
    commands = []

    def fake_run(command, cwd=None):
        command = [str(part) for part in command]
        commands.append(command)
        if any("benchmark_beit3_batch.py" in part for part in command):
            report_path = Path(command[command.index("--output") + 1])
            workers = [
                {
                    "status": "ok",
                    "samples_per_second": 2.0,
                    "peak_allocated_gib": 10.0,
                    "peak_reserved_gib": 11.0,
                    "device_total_gib": 16.0,
                }
                for _ in range(2)
            ]
            report_path.write_text(
                json.dumps(
                    {
                        "world_size": 2,
                        "image_size": 480,
                        "target_global_batch_size": 128,
                        "selected_batch_size_per_gpu": 2,
                        "results": [{"batch_size_per_gpu": 2, "status": "ok", "per_gpu": workers}],
                    }
                ),
                encoding="utf-8",
            )
        elif any("prepare_beit3_runtime_config.py" in part for part in command):
            prepare_runtime_config(
                repo_root / command[command.index("--config") + 1],
                command[command.index("--benchmark") + 1],
                command[command.index("--output") + 1],
                expected_world_size=int(command[command.index("--world-size") + 1]),
            )

    monkeypatch.setattr(run_kaggle_finetune, "run", fake_run)
    launcher = ["python", "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train.py"]

    config_path = run_kaggle_finetune.prepare_beit3_runtime(launcher)

    assert Path(config_path).is_file()
    assert [next((part for part in command if part.endswith(".py")), "") for command in commands] == [
        "scripts/benchmark_beit3_batch.py",
        "scripts/prepare_beit3_runtime_config.py",
        "scripts/smoke_beit3_distributed.py",
    ]
    assert "--nproc_per_node=2" in commands[0]
    assert "--expected-gpus" in commands[2]
    smoke = commands[2]
    batch = int(smoke[smoke.index("--batch-size-per-gpu") + 1])
    accumulation = int(smoke[smoke.index("--gradient-accumulation-steps") + 1])
    assert batch == 2
    assert batch * accumulation * 2 == 128
    assert smoke[smoke.index("--question-length") + 1] == "64"
    assert smoke[smoke.index("--answer-vocab-size") + 1] == "5000"
    commands.clear()

    run_kaggle_finetune.prepare_beit3_runtime(launcher)

    assert len(commands) == 1
    assert "scripts/smoke_beit3_distributed.py" in commands[0]


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


def _write_split_package(root: Path, split: str, image_id: int) -> Path:
    spec = run_kaggle_finetune.PACKAGED_SPLITS[split]
    image_prefix = "train" if split == "train2014" else "val"
    image_name = f"COCO_{image_prefix}2014_{image_id:012d}.jpg"
    image_source = root / "source" / image_name
    image_source.parent.mkdir(parents=True, exist_ok=True)
    image_source.write_bytes(b"jpeg-data")

    json_sources = []
    for name in spec["json_names"]:
        source = root / "source" / name
        if "_questions" in name:
            source.write_text(json.dumps({"questions": [{"image_id": image_id}]}), encoding="utf-8")
        else:
            source.write_text(json.dumps({"annotations": []}), encoding="utf-8")
        json_sources.append(source)

    output_dir = root / "notebooks" / "owner" / f"{split}-pack"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / spec["archive"]
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(image_source, arcname=f"{split}/{image_name}", recursive=False)
        for source in json_sources:
            archive.add(source, arcname=source.name, recursive=False)

    source_bytes = image_source.stat().st_size + sum(path.stat().st_size for path in json_sources)
    manifest = {
        "schema_version": 1,
        "split": split,
        "image_count": 1,
        "member_count": 3,
        "source_bytes": source_bytes,
        "json_names": list(spec["json_names"]),
        "archive_name": spec["archive"],
        "archive_size_bytes": archive_path.stat().st_size,
        "created_at": "test",
    }
    (output_dir / spec["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    return archive_path


def test_normalize_vqa_data_from_sharded_kernel_outputs(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "input" / "notebooks"
    _write_split_package(source_root, "train2014", 9)
    _write_split_package(source_root, "val2014", 42)
    extracted_root = tmp_path / "scratch" / "coco"
    work_root = tmp_path / "working"
    normalized_root = work_root / "vqa"
    monkeypatch.setattr(run_kaggle_finetune, "KERNEL_OUTPUT_ROOT", source_root)
    monkeypatch.setattr(run_kaggle_finetune, "WORK_ROOT", work_root)
    monkeypatch.setattr(run_kaggle_finetune, "PACKED_DATA_ROOT", extracted_root)
    monkeypatch.setattr(run_kaggle_finetune, "NORMALIZED_DATA_ROOT", normalized_root)
    expected_normalized_root = extracted_root / "normalized"

    result = run_kaggle_finetune.normalize_vqa_data()

    assert result == expected_normalized_root
    assert (expected_normalized_root / "train2014" / "COCO_train2014_000000000009.jpg").is_file()
    assert (expected_normalized_root / "val2014" / "COCO_val2014_000000000042.jpg").is_file()
    assert (expected_normalized_root / "v2_OpenEnded_mscoco_train2014_questions.json").is_file()
    assert (expected_normalized_root / "v2_mscoco_val2014_annotations.json").is_file()
    assert not normalized_root.exists()
    assert (extracted_root / ".vqa-package.json").is_file()


def test_normalize_vqa_data_downloads_coco_when_no_input_is_mounted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "official-coco"
    train_images = source_root / "train2014"
    val_images = source_root / "val2014"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "COCO_train2014_000000000009.jpg").write_bytes(b"train")
    (val_images / "COCO_val2014_000000000042.jpg").write_bytes(b"val")
    questions = {
        "v2_OpenEnded_mscoco_train2014_questions.json": {
            "questions": [{"image_id": 9}]
        },
        "v2_OpenEnded_mscoco_val2014_questions.json": {
            "questions": [{"image_id": 42}]
        },
    }
    annotation_names = {
        "v2_mscoco_train2014_annotations.json",
        "v2_mscoco_val2014_annotations.json",
    }
    downloaded_files = {}
    for name, payload in questions.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        downloaded_files[name] = path
    for name in annotation_names:
        path = tmp_path / name
        path.write_text(json.dumps({"annotations": []}), encoding="utf-8")
        downloaded_files[name] = path

    work_root = tmp_path / "working"
    normalized_root = work_root / "vqa"
    monkeypatch.setattr(run_kaggle_finetune, "KERNEL_OUTPUT_ROOT", tmp_path / "no-input")
    monkeypatch.setattr(run_kaggle_finetune, "WORK_ROOT", work_root)
    monkeypatch.setattr(run_kaggle_finetune, "NORMALIZED_DATA_ROOT", normalized_root)
    monkeypatch.setattr(
        run_kaggle_finetune,
        "resolve_vqa_data_root",
        lambda _candidates: (_ for _ in ()).throw(FileNotFoundError("no mounted data")),
    )
    monkeypatch.setattr(
        run_kaggle_finetune,
        "download_coco_images",
        lambda: (source_root, train_images, val_images),
    )
    monkeypatch.setattr(
        run_kaggle_finetune,
        "download_vqa_file",
        lambda name: downloaded_files[name],
    )

    result = run_kaggle_finetune.normalize_vqa_data()

    assert result == normalized_root
    assert (normalized_root / "train2014" / "COCO_train2014_000000000009.jpg").is_file()
    assert (normalized_root / "val2014" / "COCO_val2014_000000000042.jpg").is_file()
    assert (normalized_root / "v2_mscoco_train2014_annotations.json").is_file()


def test_packaged_input_smoke_validates_referenced_images_and_cuda(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "input" / "notebooks"
    _write_split_package(source_root, "train2014", 9)
    _write_split_package(source_root, "val2014", 42)
    monkeypatch.setattr(run_kaggle_finetune, "KERNEL_OUTPUT_ROOT", source_root)
    monkeypatch.setattr(run_kaggle_finetune, "PACKED_DATA_ROOT", tmp_path / "scratch" / "coco")
    monkeypatch.setattr(run_kaggle_finetune, "NORMALIZED_DATA_ROOT", tmp_path / "working" / "vqa")
    launcher_calls = []
    run_calls = []
    monkeypatch.setattr(
        run_kaggle_finetune,
        "training_launcher",
        lambda cwd: launcher_calls.append(cwd),
    )
    monkeypatch.setattr(
        run_kaggle_finetune,
        "run",
        lambda command, cwd=None: run_calls.append((command, cwd)),
    )

    run_kaggle_finetune.run_packaged_input_smoke()

    assert launcher_calls == [Path.cwd()]
    assert len(run_calls) == 1
    assert "torch.cuda.device_count()" in run_calls[0][0][2]
    assert run_calls[0][1] == Path.cwd()


def test_sharded_vqa_input_requires_both_splits(tmp_path: Path) -> None:
    _write_split_package(tmp_path / "source", "train2014", 9)

    with pytest.raises(FileNotFoundError, match=r"missing package\(s\): val2014"):
        run_kaggle_finetune.find_packaged_vqa_archives(tmp_path / "source" / "notebooks")


def test_kaggle_tar_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar"
    member_path = tmp_path / "payload.txt"
    member_path.write_text("unsafe", encoding="utf-8")
    with tarfile.open(archive_path, mode="w") as archive:
        archive.add(member_path, arcname="../outside.txt", recursive=False)

    manifest = {
        "split": "train2014",
        "json_names": [],
        "member_count": 1,
        "image_count": 1,
        "source_bytes": member_path.stat().st_size,
    }
    output_root = tmp_path / "extract"
    with pytest.raises(tarfile.ReadError, match="Unsafe or unsupported"):
        run_kaggle_finetune.extract_vqa_tar(archive_path, output_root, manifest)
    assert not (tmp_path / "extract" / "outside.txt").exists()


def test_kaggle_download_rejects_non_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Download URL"):
        run_kaggle_finetune.download_https("file:///tmp/data.zip", tmp_path / "data.zip")


def test_resumable_download_appends_from_server_range(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "train2014.zip"
    output.with_name(output.name + ".part").write_bytes(b"abc")
    requests = []

    class FakeResponse(io.BytesIO):
        status = 206
        headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(b"def")

    monkeypatch.setattr(run_kaggle_finetune.urllib.request, "urlopen", fake_urlopen)

    result = run_kaggle_finetune.download_resumable_https(
        "https://example.com/train2014.zip",
        output,
        max_attempts=1,
    )

    assert result == output
    assert output.read_bytes() == b"abcdef"
    assert requests[0][0].get_header("Range") == "bytes=3-"
    assert requests[0][1] == 90
    assert not output.with_name(output.name + ".part").exists()


def test_probe_coco_archives_uses_https_head_requests(monkeypatch) -> None:
    requests = []

    class FakeResponse(io.BytesIO):
        status = 200
        headers = {"Content-Length": "1234"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(b"")

    monkeypatch.setattr(run_kaggle_finetune.urllib.request, "urlopen", fake_urlopen)

    sizes = run_kaggle_finetune.probe_coco_archive_sources()

    assert sizes == {"train2014": 1234, "val2014": 1234}
    assert len(requests) == 2
    assert all(request.get_method() == "HEAD" for request, _ in requests)
    assert all(timeout == 30 for _, timeout in requests)


def test_download_coco_images_extracts_and_reuses_official_archives(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "coco"
    urls = []
    monkeypatch.setattr(run_kaggle_finetune, "COCO_DOWNLOAD_ROOT", root)
    monkeypatch.setattr(run_kaggle_finetune, "COCO_IMAGE_COUNTS", {"train2014": 1, "val2014": 1})

    def fake_download(url, output):
        urls.append(url)
        split = Path(url).stem
        prefix = "train" if split == "train2014" else "val"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                f"{split}/COCO_{prefix}2014_000000000001.jpg",
                b"jpeg-data",
            )
        return output

    monkeypatch.setattr(run_kaggle_finetune, "download_resumable_https", fake_download)

    result = run_kaggle_finetune.download_coco_images()
    cached_result = run_kaggle_finetune.download_coco_images()

    assert result == (root, root / "train2014", root / "val2014")
    assert cached_result == result
    assert urls == [
        f"{run_kaggle_finetune.COCO_ARCHIVE_BASE_URL}/train2014.zip",
        f"{run_kaggle_finetune.COCO_ARCHIVE_BASE_URL}/val2014.zip",
    ]
    assert len(run_kaggle_finetune.available_image_ids(root / "train2014")) == 1
    assert len(run_kaggle_finetune.available_image_ids(root / "val2014")) == 1
    assert not (root / "train2014.zip").exists()
    assert not (root / "val2014.zip").exists()


def test_resolve_resume_root_finds_owner_scoped_kaggle_dataset_mount(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    dataset_root = (
        input_root
        / "datasets"
        / run_kaggle_finetune.RESUME_DATASET_OWNER
        / run_kaggle_finetune.RESUME_DATASET_SLUG
    )
    dataset_root.mkdir(parents=True)
    (dataset_root / "latest.pt").touch()
    monkeypatch.setattr(run_kaggle_finetune, "RESUME_ROOT", tmp_path / "not-mounted")

    assert run_kaggle_finetune.resolve_resume_root(input_root) == dataset_root
