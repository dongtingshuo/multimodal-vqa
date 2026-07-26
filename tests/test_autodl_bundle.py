from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripts.build_autodl_bundle import extract_repository_archive
from vqa_project.config import load_config
from vqa_project.training import resume_signature


def test_autodl_config_preserves_resume_signature() -> None:
    original = load_config("configs/kaggle_vilt.yaml")
    autodl = load_config("autodl/configs/vilt_resume_4090d.yaml")

    assert resume_signature(autodl) == resume_signature(original)
    assert autodl["data"]["num_workers"] == 8
    assert autodl["train"]["batch_size"] == 4
    assert autodl["train"]["gradient_accumulation_steps"] == 8
    assert autodl["tracking"]["wandb"]["enabled"] is False


def test_autodl_pipeline_requires_format_v3_resume() -> None:
    pipeline = Path("autodl/run_pipeline.sh").read_text(encoding="utf-8")
    setup = Path("autodl/setup.sh").read_text(encoding="utf-8")

    assert '--resume "${RUN_DIR}/latest.pt"' in pipeline
    assert "--epochs 10" in pipeline
    assert "--no-wandb" in pipeline
    assert "-m autodl.preflight" in setup
    assert "latest.pt" in setup and "best.pt" in setup


def test_repository_archive_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar"
    payload = b"unsafe"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.ReadError, match="Unsafe archive member"):
        extract_repository_archive(archive_path, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()


def test_repository_archive_extracts_regular_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "repository.tar"
    payload = b"print('ok')\n"
    with tarfile.open(archive_path, "w") as archive:
        directory = tarfile.TarInfo("project")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo("project/app.py")
        member.mode = 0o755
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    target = tmp_path / "output"
    extract_repository_archive(archive_path, target)
    assert (target / "project" / "app.py").read_bytes() == payload
    assert (target / "project" / "app.py").stat().st_mode & 0o111
