from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "multimodal-vqa-autodl"
RESUME_FILES = (
    "latest.pt",
    "training_history.csv",
    "training_curves.png",
    "config.snapshot.json",
    "run_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AutoDL ViLT continuation upload bundle.")
    parser.add_argument("--resume-dir", required=True, help="Directory containing epoch-2 latest.pt and run artifacts.")
    parser.add_argument("--answer-vocab", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def export_repository(target: Path) -> str:
    commit = git("rev-parse", "HEAD")
    archive_path = target.parent / "repository.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", archive_path, commit],
        cwd=PROJECT_ROOT,
        check=True,
    )
    target.mkdir(parents=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(target)
    archive_path.unlink()
    return commit


def build_bundle(resume_dir: Path, answer_vocab: Path, output: Path) -> None:
    latest = resume_dir / "latest.pt"
    if not latest.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {latest}")
    if not answer_vocab.is_file():
        raise FileNotFoundError(f"Missing answer vocabulary: {answer_vocab}")

    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    if int(checkpoint.get("format_version", 0)) < 3 or int(checkpoint.get("epoch", -1)) != 2:
        raise ValueError("AutoDL bundle requires the format-v3 epoch-2 latest.pt checkpoint.")
    if "optimizer_state" not in checkpoint or len(checkpoint.get("idx_to_answer") or []) != 3000:
        raise ValueError("Resume checkpoint is missing optimizer state or the Top-3000 answer vocabulary.")
    del checkpoint

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="multimodal-vqa-autodl-") as temporary:
        staging = Path(temporary) / PACKAGE_NAME
        repo_target = staging / "repo"
        resume_target = staging / "resume" / "vilt-seed42"
        resume_target.mkdir(parents=True)
        commit = export_repository(repo_target)

        shutil.copy2(answer_vocab, staging / "resume" / "answer_vocab.json")
        copied = []
        for name in RESUME_FILES:
            source = resume_dir / name
            if source.is_file():
                target = resume_target / name
                shutil.copy2(source, target)
                copied.append(target)

        manifest_paths = [staging / "resume" / "answer_vocab.json", *copied]
        manifest_lines = [f"{sha256(path)}  {path.relative_to(staging)}" for path in manifest_paths]
        (staging / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        bundle_info = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "checkpoint_epoch": 2,
            "resume_epoch": 3,
            "checkpoint_sha256": sha256(resume_target / "latest.pt"),
            "checkpoint_source": "resume/vilt-seed42/latest.pt",
            "target": "AutoDL RTX 4090D/4090, PyTorch 2.5.1, Python 3.12, CUDA 12.4",
        }
        (staging / "BUNDLE_INFO.json").write_text(
            json.dumps(bundle_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output.unlink(missing_ok=True)
        with tarfile.open(output, "w") as archive:
            archive.add(staging, arcname=PACKAGE_NAME)

    print(json.dumps({"output": str(output), "size": output.stat().st_size, **bundle_info}, indent=2))


def main() -> None:
    args = parse_args()
    build_bundle(Path(args.resume_dir), Path(args.answer_vocab), Path(args.output))


if __name__ == "__main__":
    main()
