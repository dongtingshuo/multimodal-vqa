import json
import re
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

SPLIT = "train2014"
MIN_IMAGES = 80_000
ARCHIVE_NAME = "coco_train_vqa.tar"
MANIFEST_NAME = "coco_train_vqa_manifest.json"
JSON_NAMES = (
    "v2_OpenEnded_mscoco_train2014_questions.json",
    "v2_mscoco_train2014_annotations.json",
)
DATA_ROOT = Path(
    "/kaggle/input/datasets/sagnikkayalcse52/coco2014vqa/Dataset"
)
OUTPUT_ROOT = Path("/kaggle/working")
MAX_OUTPUT_BYTES = 19 * 1024**3
IMAGE_PATTERN = re.compile(r"COCO_train2014_\d{12}\.jpg$")


def emit(message):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"VQA_DATA_PACK {timestamp} {message}", flush=True)


def main():
    image_dir = DATA_ROOT / "images" / SPLIT
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing COCO image directory: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and IMAGE_PATTERN.fullmatch(path.name)
    )
    if len(image_paths) < MIN_IMAGES:
        raise RuntimeError(
            f"Refusing to package an incomplete {SPLIT}: "
            f"found {len(image_paths):,}, require at least {MIN_IMAGES:,} images"
        )

    json_paths = [DATA_ROOT / "json" / name for name in JSON_NAMES]
    missing_json = [str(path) for path in json_paths if not path.is_file()]
    if missing_json:
        raise FileNotFoundError("Missing VQA annotation files: " + ", ".join(missing_json))

    source_bytes = sum(path.stat().st_size for path in image_paths + json_paths)
    member_count = len(image_paths) + len(json_paths)
    estimated_tar_bytes = source_bytes + member_count * 1024 + 10_240
    if estimated_tar_bytes > MAX_OUTPUT_BYTES:
        raise RuntimeError(
            f"Estimated archive size {estimated_tar_bytes:,} exceeds the safe "
            f"per-kernel output budget {MAX_OUTPUT_BYTES:,} bytes"
        )
    available_bytes = shutil.disk_usage(OUTPUT_ROOT).free
    if estimated_tar_bytes > available_bytes:
        raise OSError(
            f"Not enough output space: need about {estimated_tar_bytes:,} bytes, "
            f"have {available_bytes:,}"
        )

    archive_path = OUTPUT_ROOT / ARCHIVE_NAME
    temporary_path = archive_path.with_suffix(".tar.partial")
    temporary_path.unlink(missing_ok=True)
    started = time.monotonic()
    emit(
        f"START split={SPLIT} images={len(image_paths):,} "
        f"source_bytes={source_bytes:,} output={archive_path}"
    )

    with tarfile.open(temporary_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        archive.copybufsize = 1024 * 1024
        for index, path in enumerate(image_paths, start=1):
            archive.add(path, arcname=f"{SPLIT}/{path.name}", recursive=False)
            if index % 5_000 == 0 or index == len(image_paths):
                emit(f"PROGRESS split={SPLIT} images={index:,}/{len(image_paths):,}")
        for path in json_paths:
            archive.add(path, arcname=path.name, recursive=False)

    archive_size = temporary_path.stat().st_size
    if archive_size > MAX_OUTPUT_BYTES:
        temporary_path.unlink()
        raise RuntimeError(
            f"Created archive is too large for a Kaggle output: {archive_size:,} bytes"
        )
    temporary_path.replace(archive_path)

    manifest = {
        "schema_version": 1,
        "split": SPLIT,
        "image_count": len(image_paths),
        "member_count": member_count,
        "source_bytes": source_bytes,
        "json_names": list(JSON_NAMES),
        "archive_name": ARCHIVE_NAME,
        "archive_size_bytes": archive_size,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUTPUT_ROOT / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    emit(
        f"COMPLETE split={SPLIT} archive_bytes={archive_size:,} "
        f"elapsed_seconds={time.monotonic() - started:.1f}"
    )


if __name__ == "__main__":
    main()
