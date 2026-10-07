import codecs
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working/multimodal-vqa"))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/kaggle/working/multimodal-vqa-repo"))
RUN_NAME = os.environ.get("RUN_NAME", "vilt-last6-t4x2")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/kaggle_vilt_last6_t4x2.yaml")
GIT_REF = os.environ.get("GIT_REF", "main")
TOTAL_EPOCHS = os.environ.get("TOTAL_EPOCHS", "10")
VQA_NUM_GPUS = os.environ.get("VQA_NUM_GPUS", "2").strip().lower()
DATA_SMOKE_ONLY = os.environ.get("VQA_DATA_SMOKE_ONLY", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
NETWORK_SMOKE_ONLY = os.environ.get("VQA_NETWORK_SMOKE_ONLY", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
RAW_DATA_ROOT = Path(
    os.environ.get(
        "RAW_DATA_ROOT",
        "/kaggle/input/datasets/sagnikkayalcse52/coco2014vqa/Dataset",
    )
)
KERNEL_OUTPUT_ROOT = Path("/kaggle/input/notebooks")
PACKED_DATA_ROOT = Path(
    os.environ.get("COCO_PACKED_DATA_ROOT", "/tmp/multimodal-vqa-coco-pack-v1")
)
COCO_DOWNLOAD_ROOT = Path(
    os.environ.get("COCO_DOWNLOAD_ROOT", "/tmp/multimodal-vqa-coco-direct")
)
COCO_ARCHIVE_BASE_URL = os.environ.get(
    "COCO_ARCHIVE_BASE_URL", "https://s3.amazonaws.com/images.cocodataset.org/zips"
)
COCO_IMAGE_COUNTS = {"train2014": 82_783, "val2014": 40_504}
PACKAGED_SPLITS = {
    "train2014": {
        "archive": "coco_train_vqa.tar",
        "manifest": "coco_train_vqa_manifest.json",
        "json_names": (
            "v2_OpenEnded_mscoco_train2014_questions.json",
            "v2_mscoco_train2014_annotations.json",
        ),
    },
    "val2014": {
        "archive": "coco_val_vqa.tar",
        "manifest": "coco_val_vqa_manifest.json",
        "json_names": (
            "v2_OpenEnded_mscoco_val2014_questions.json",
            "v2_mscoco_val2014_annotations.json",
        ),
    },
}
RESUME_ROOT = Path(
    os.environ.get("RESUME_ROOT", "/kaggle/input/multimodal-vqa-vilt-last6-t4x2-resume")
)
TORCH_VERSION = os.environ.get("TORCH_VERSION", "2.5.1+cu121")
TORCHVISION_VERSION = os.environ.get("TORCHVISION_VERSION", "0.20.1+cu121")
TRANSFORMERS_SPEC = os.environ.get("TRANSFORMERS_SPEC", "transformers>=5.10,<6.0")
PYTORCH_INDEX_URL = os.environ.get("PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")
PYTORCH_RUNTIME_DIR = Path(
    os.environ.get("PYTORCH_RUNTIME_DIR", WORK_ROOT / "pytorch-runtime")
)
COCO_IMAGE_BASE_URL = os.environ.get(
    "COCO_IMAGE_BASE_URL", "https://s3.amazonaws.com/images.cocodataset.org"
)

CHECKPOINT_DIR = WORK_ROOT / RUN_NAME
ANSWER_VOCAB = WORK_ROOT / "answer_vocab.json"
PREDICTIONS_PATH = CHECKPOINT_DIR / "val_predictions.json"
OFFICIAL_METRICS_PATH = CHECKPOINT_DIR / "official_vqa_metrics.json"
VQA_TOOLKIT_ROOT = WORK_ROOT / "official-vqa-toolkit"
ARCHIVE_PATH = Path("/kaggle/working/multimodal-vqa-finetune-artifacts")
EXPORT_ROOT = Path("/kaggle/working/multimodal-vqa-export")
NORMALIZED_DATA_ROOT = Path(os.environ.get("DATA_ROOT", WORK_ROOT / "vqa"))
DOWNLOAD_ROOT = WORK_ROOT / "downloads"
RESUME_FILES = (
    "latest.pt",
    "best.pt",
    "config.snapshot.json",
    "training_history.csv",
    "training_curves.png",
    "run_metadata.json",
    "run_summary.json",
)

VQA_DOWNLOADS = {
    "v2_OpenEnded_mscoco_train2014_questions.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_mscoco_train2014_annotations.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_OpenEnded_mscoco_val2014_questions.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_mscoco_val2014_annotations.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}
COCO_IMAGE_RE = re.compile(r"COCO_(?:train|val)2014_(\d{12})\.jpg$")
HEARTBEAT_SECONDS = 60
RUN_LOG_PATH = None
RUN_STATUS_PATH = None
RUN_STARTED_AT = None
LAST_ACTIVITY_AT = 0.0
CURRENT_STAGE = "startup"
PROGRESS_LOCK = threading.Lock()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_status(state: str, stage: str, message: str) -> None:
    if RUN_STATUS_PATH is None:
        return
    payload = {
        "state": state,
        "stage": stage,
        "message": message,
        "updated_at": _timestamp(),
        "elapsed_seconds": round(time.monotonic() - RUN_STARTED_AT, 1) if RUN_STARTED_AT else 0.0,
        "run_name": RUN_NAME,
        "config": CONFIG_PATH,
        "requested_gpus": VQA_NUM_GPUS,
    }
    temporary = RUN_STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(RUN_STATUS_PATH)


def progress(message: str, *, state: str = "running", stage: Optional[str] = None) -> None:
    global LAST_ACTIVITY_AT
    stage = stage or CURRENT_STAGE
    line = f"[{_timestamp()}] [{stage}] {message}"
    with PROGRESS_LOCK:
        LAST_ACTIVITY_AT = time.monotonic()
        print(line, flush=True)
        if RUN_LOG_PATH is not None:
            with RUN_LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
                log_file.flush()
        _write_status(state, stage, message)


def initialize_progress_logging() -> None:
    global RUN_LOG_PATH, RUN_STATUS_PATH, RUN_STARTED_AT, LAST_ACTIVITY_AT
    RUN_LOG_PATH = CHECKPOINT_DIR / "runner.log"
    RUN_STATUS_PATH = CHECKPOINT_DIR / "runner_status.json"
    RUN_STARTED_AT = time.monotonic()
    LAST_ACTIVITY_AT = RUN_STARTED_AT
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text("", encoding="utf-8")
    progress("Kaggle runner process started; preparing the first preflight.", stage="bootstrap")


@contextmanager
def progress_stage(name: str):
    global CURRENT_STAGE, LAST_ACTIVITY_AT
    previous_stage = CURRENT_STAGE
    CURRENT_STAGE = name
    started = time.monotonic()
    progress("stage started")
    heartbeat_stop = threading.Event()

    def report_if_idle() -> None:
        while not heartbeat_stop.wait(HEARTBEAT_SECONDS):
            with PROGRESS_LOCK:
                idle_for = time.monotonic() - LAST_ACTIVITY_AT
            if idle_for >= HEARTBEAT_SECONDS:
                progress(
                    f"stage still running; elapsed={time.monotonic() - started:.0f}s, "
                    f"no log activity for {idle_for:.0f}s"
                )

    heartbeat = threading.Thread(target=report_if_idle, daemon=True)
    heartbeat.start()
    failed = False
    try:
        yield
    except BaseException as exc:
        failed = True
        progress(f"stage failed after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}", state="failed")
        raise
    else:
        progress(f"stage completed in {time.monotonic() - started:.1f}s")
    finally:
        heartbeat_stop.set()
        heartbeat.join()
        if not failed:
            CURRENT_STAGE = previous_stage


def _forward_process_output(stream, last_output: list[float]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    while True:
        chunk = os.read(stream.fileno(), 4096)
        if not chunk:
            break
        last_output[0] = time.monotonic()
        text = decoder.decode(chunk)
        if not text:
            continue
        with PROGRESS_LOCK:
            global LAST_ACTIVITY_AT
            LAST_ACTIVITY_AT = time.monotonic()
            sys.stdout.write(text)
            sys.stdout.flush()
            if RUN_LOG_PATH is not None:
                with RUN_LOG_PATH.open("a", encoding="utf-8") as log_file:
                    log_file.write(text.replace("\r", "\n"))
                    log_file.flush()
    tail = decoder.decode(b"", final=True)
    if tail:
        with PROGRESS_LOCK:
            LAST_ACTIVITY_AT = time.monotonic()
            sys.stdout.write(tail)
            sys.stdout.flush()
            if RUN_LOG_PATH is not None:
                with RUN_LOG_PATH.open("a", encoding="utf-8") as log_file:
                    log_file.write(tail.replace("\r", "\n"))
                    log_file.flush()


def run(command, cwd=None):
    command = [str(part) for part in command]
    command_text = " ".join(command)
    progress(f"command started: {command_text}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    last_output = [time.monotonic()]
    reader = threading.Thread(target=_forward_process_output, args=(process.stdout, last_output), daemon=True)
    reader.start()
    while True:
        try:
            return_code = process.wait(timeout=HEARTBEAT_SECONDS)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            silent_for = time.monotonic() - last_output[0]
            if silent_for >= HEARTBEAT_SECONDS:
                progress(f"command still running; elapsed={elapsed:.0f}s, no child output for {silent_for:.0f}s")
    reader.join()
    elapsed = time.monotonic() - started
    if return_code:
        progress(f"command failed with exit code {return_code} after {elapsed:.1f}s: {command_text}", state="failed")
        raise subprocess.CalledProcessError(return_code, command)
    progress(f"command completed in {elapsed:.1f}s: {command_text}")


def training_launcher(cwd=REPO_ROOT):
    progress(f"checking visible CUDA devices; requested GPU count={VQA_NUM_GPUS}")
    probe = subprocess.run(
        ["python", "-c", "import torch; print(torch.cuda.device_count())"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    available = int(probe.stdout.strip().splitlines()[-1])
    requested = available if VQA_NUM_GPUS == "auto" else int(VQA_NUM_GPUS)
    if requested < 1 or requested > available:
        raise ValueError(
            f"VQA_NUM_GPUS={requested} requested, but this Kaggle session exposes only {available} CUDA GPU(s). "
            "Select GPU T4 x2 for this run, or set VQA_NUM_GPUS=1 to intentionally use one GPU."
        )
    progress(f"Using {requested} of {available} visible GPU(s) for training")
    if requested == 1:
        return ["python", "-u", "train.py"]
    return [
        "python",
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={requested}",
        "train.py",
    ]


def torch_runtime_is_usable():
    progress("checking the installed PyTorch build with a real CUDA tensor operation")
    probe = subprocess.run(
        [
            "python",
            "-c",
            (
                "import torch, torchvision; "
                "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                "value = torch.ones(1, device='cuda'); "
                "torch.cuda.synchronize(); "
                "assert value.item() == 1.0, 'CUDA tensor probe failed'; "
                "print(f'Using preinstalled torch={torch.__version__} '"
                "f'torchvision={torchvision.__version__} cuda={torch.version.cuda} '"
                "f'device_capability={torch.cuda.get_device_capability()}')"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if probe.stdout.strip():
        progress(probe.stdout.strip())
    if probe.returncode == 0:
        return True
    if probe.stderr.strip():
        progress(f"Preinstalled PyTorch probe failed: {probe.stderr.strip()}")
    return False


def preinstalled_torch_is_usable():
    if os.environ.get("FORCE_TORCH_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
        progress("FORCE_TORCH_INSTALL is enabled; installing the pinned PyTorch stack")
        return False
    return torch_runtime_is_usable()


def activate_pinned_torch_runtime():
    PYTORCH_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--upgrade",
            "--target",
            PYTORCH_RUNTIME_DIR,
            "--index-url",
            PYTORCH_INDEX_URL,
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
        ],
        cwd=REPO_ROOT,
    )
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [str(PYTORCH_RUNTIME_DIR)]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    if not torch_runtime_is_usable():
        raise RuntimeError(f"Pinned PyTorch runtime failed its CUDA probe: {PYTORCH_RUNTIME_DIR}")


def install_training_dependencies():
    if not preinstalled_torch_is_usable():
        activate_pinned_torch_runtime()
    run(
        [
            "python",
            "-m",
            "pip",
            "install",
            TRANSFORMERS_SPEC,
            "Pillow>=10.0",
            "PyYAML>=6.0",
            "tqdm>=4.66",
            "matplotlib>=3.8",
        ],
        cwd=REPO_ROOT,
    )


def find_dir(root, name):
    direct = root / name
    if direct.is_dir():
        return direct
    for path in root.rglob(name):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find directory {name!r} under {root}")


def resolve_vqa_data_root(candidates):
    searched = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        searched.append(str(candidate))
        try:
            train_images = find_dir(candidate, "train2014")
            val_images = find_dir(candidate, "val2014")
        except FileNotFoundError:
            continue
        return candidate, train_images, val_images
    raise FileNotFoundError(
        "Could not find both train2014 and val2014 image directories. "
        "Checked input roots: " + (", ".join(searched) if searched else "none exist")
    )


def find_file(root, name):
    direct = root / name
    if direct.is_file():
        return direct
    for path in root.rglob(name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find file {name!r} under {root}")


def _find_packaged_split(input_root, split):
    spec = PACKAGED_SPLITS[split]
    if not input_root.is_dir():
        return None

    archives = list(input_root.rglob(spec["archive"]))
    manifests = list(input_root.rglob(spec["manifest"]))
    if not archives and not manifests:
        return None
    if len(archives) != 1 or len(manifests) != 1:
        raise ValueError(
            f"Expected exactly one {split} archive and manifest under {input_root}; "
            f"found {len(archives)} archive(s) and {len(manifests)} manifest(s)"
        )

    archive_path = archives[0]
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("split") != split:
        raise ValueError(f"Unsupported or mismatched {split} package manifest: {manifest_path}")
    if manifest.get("archive_name") != spec["archive"]:
        raise ValueError(f"Archive name mismatch in package manifest: {manifest_path}")
    if manifest.get("archive_size_bytes") != archive_path.stat().st_size:
        raise ValueError(f"Archive size does not match package manifest: {archive_path}")
    if tuple(manifest.get("json_names", ())) != spec["json_names"]:
        raise ValueError(f"Annotation manifest mismatch: {manifest_path}")
    image_count = manifest.get("image_count")
    source_bytes = manifest.get("source_bytes")
    if not isinstance(image_count, int) or image_count < 1:
        raise ValueError(f"Invalid image count in package manifest: {manifest_path}")
    if not isinstance(source_bytes, int) or source_bytes < 1:
        raise ValueError(f"Invalid source size in package manifest: {manifest_path}")
    if manifest.get("member_count") != image_count + len(spec["json_names"]):
        raise ValueError(f"Invalid member count in package manifest: {manifest_path}")
    return {"archive": archive_path, "manifest": manifest}


def find_packaged_vqa_archives(input_root=KERNEL_OUTPUT_ROOT):
    packaged = {
        split: _find_packaged_split(input_root, split)
        for split in PACKAGED_SPLITS
    }
    if all(package is None for package in packaged.values()):
        return None
    missing = [split for split, package in packaged.items() if package is None]
    if missing:
        raise FileNotFoundError(
            "Incomplete sharded COCO inputs; missing package(s): " + ", ".join(missing)
        )
    return packaged


def extract_vqa_tar(archive_path, target_root, manifest):
    target_root = Path(target_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    split = manifest["split"]
    expected_json = set(manifest["json_names"])
    seen_names = set()
    image_count = 0
    extracted_bytes = 0

    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) != manifest["member_count"]:
            raise tarfile.ReadError(
                f"{archive_path} has {len(members)} members; "
                f"manifest declares {manifest['member_count']}"
            )
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (
                not parts
                or member.name.startswith("/")
                or "\\" in member.name
                or ".." in parts
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise tarfile.ReadError(f"Unsafe or unsupported archive member: {member.name}")

            if len(parts) == 2 and parts[0] == split and COCO_IMAGE_RE.fullmatch(parts[1]):
                image_count += 1
            elif len(parts) == 1 and parts[0] in expected_json:
                pass
            else:
                raise tarfile.ReadError(f"Unexpected {split} archive member: {member.name}")
            if member.name in seen_names:
                raise tarfile.ReadError(f"Duplicate archive member: {member.name}")
            seen_names.add(member.name)

            destination = (target_root / Path(*parts)).resolve()
            if not destination.is_relative_to(target_root):
                raise tarfile.ReadError(f"Unsafe archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise tarfile.ReadError(f"Could not read archive member: {member.name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            written = destination.stat().st_size
            if written != member.size:
                raise tarfile.ReadError(f"Truncated archive member: {member.name}")
            extracted_bytes += written

    if image_count != manifest["image_count"]:
        raise tarfile.ReadError(
            f"{split} archive contains {image_count} images; "
            f"manifest declares {manifest['image_count']}"
        )
    if expected_json != {name for name in seen_names if "/" not in name}:
        raise tarfile.ReadError(f"{split} archive is missing expected VQA annotation files")
    if extracted_bytes != manifest["source_bytes"]:
        raise tarfile.ReadError(
            f"{split} archive extracted {extracted_bytes} bytes; "
            f"manifest declares {manifest['source_bytes']}"
        )
    return {"image_count": image_count, "extracted_bytes": extracted_bytes}


def extract_packaged_vqa_data(packaged):
    PACKED_DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    signatures = {
        split: {
            "archive": str(package["archive"]),
            "archive_size_bytes": package["manifest"]["archive_size_bytes"],
            "created_at": package["manifest"].get("created_at"),
        }
        for split, package in packaged.items()
    }
    marker_path = PACKED_DATA_ROOT / ".vqa-package.json"
    if PACKED_DATA_ROOT.is_dir() and marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("archives") == signatures:
            return PACKED_DATA_ROOT
        raise FileExistsError(
            f"Existing extracted data at {PACKED_DATA_ROOT} belongs to different archives"
        )
    if PACKED_DATA_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite unverified extracted data at {PACKED_DATA_ROOT}"
        )

    required_bytes = sum(package["manifest"]["source_bytes"] for package in packaged.values())
    available_bytes = shutil.disk_usage(PACKED_DATA_ROOT.parent).free
    if required_bytes > available_bytes:
        raise OSError(
            f"Not enough temporary disk space to unpack COCO: need {required_bytes:,} bytes, "
            f"have {available_bytes:,}"
        )

    staging_root = PACKED_DATA_ROOT.with_name(f".{PACKED_DATA_ROOT.name}.partial")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    try:
        for split, package in packaged.items():
            with progress_stage(f"unpack {split}"):
                result = extract_vqa_tar(
                    package["archive"], staging_root, package["manifest"]
                )
                progress(
                    f"unpacked {result['image_count']:,} {split} images "
                    f"({result['extracted_bytes']:,} bytes)"
                )
        marker = {"schema_version": 1, "archives": signatures}
        (staging_root / ".vqa-package.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        staging_root.replace(PACKED_DATA_ROOT)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return PACKED_DATA_ROOT


def validate_https_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Download URL must use HTTPS and include a host.")
    if parsed.username or parsed.password:
        raise ValueError("Download URL must not contain embedded credentials.")


def download_https(url, output):
    validate_https_url(url)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "multimodal-vqa-kaggle-runner/1.0"})
    downloaded = 0
    last_report = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as destination:
            expected = int(response.headers.get("Content-Length", "0") or 0)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 30:
                    progress(f"downloaded {downloaded:,} bytes" + (f" of {expected:,}" if expected else ""))
                    last_report = time.monotonic()
        if downloaded == 0:
            raise OSError(f"Download returned an empty response: {url}")
        if expected and downloaded != expected:
            raise OSError(f"Incomplete download: expected {expected} bytes, received {downloaded} bytes from {url}")
        temporary.replace(output)
        progress(f"download complete: {output.name} ({downloaded:,} bytes)")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def download_resumable_https(url, output, *, max_attempts=5):
    validate_https_url(url)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")

    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "multimodal-vqa-kaggle-runner/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                content_range = response.headers.get("Content-Range", "")
                if offset and status == 206:
                    match = re.fullmatch(r"bytes (\d+)-\d+/(\d+|\*)", content_range)
                    if match is None or int(match.group(1)) != offset:
                        raise OSError(f"Invalid Content-Range while resuming {url}: {content_range}")
                    expected = int(match.group(2)) if match.group(2) != "*" else None
                    mode = "ab"
                else:
                    if offset:
                        progress(f"server restarted {Path(url).name} from byte zero")
                    offset = 0
                    expected = int(response.headers.get("Content-Length", "0") or 0) or None
                    mode = "wb"

                downloaded = offset
                last_report = time.monotonic()
                with partial.open(mode) as destination:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)
                        downloaded += len(chunk)
                        if time.monotonic() - last_report >= 30:
                            progress(
                                f"downloading {Path(url).name}: {downloaded:,} bytes"
                                + (f" of {expected:,}" if expected else "")
                            )
                            last_report = time.monotonic()

            actual = partial.stat().st_size
            if expected and actual != expected:
                raise OSError(
                    f"Incomplete download for {url}: expected {expected:,} bytes, "
                    f"received {actual:,}"
                )
            if actual == 0:
                raise OSError(f"Download returned an empty response: {url}")
            partial.replace(output)
            progress(f"download complete: {output.name} ({actual:,} bytes)")
            return output
        except Exception as exc:
            progress(f"download attempt {attempt}/{max_attempts} failed for {output.name}: {exc}")
            if attempt == max_attempts:
                raise
            time.sleep(min(2**attempt, 30))

    raise OSError(f"Download failed after {max_attempts} attempts: {url}")


def probe_coco_archive_sources():
    sizes = {}
    for split in COCO_IMAGE_COUNTS:
        url = f"{COCO_ARCHIVE_BASE_URL}/{split}.zip"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "multimodal-vqa-kaggle-runner/1.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            size = int(response.headers.get("Content-Length", "0") or 0)
        if status != 200 or size <= 0:
            raise OSError(f"COCO archive source check failed for {url}: status={status}, size={size}")
        sizes[split] = size
        progress(f"official COCO source reachable: {split}.zip ({size:,} bytes)")
    return sizes


def _archive_image_count(archive, split):
    count = 0
    for member in archive.infolist():
        if member.is_dir() and PurePosixPath(member.filename).parts == (split,):
            continue
        parts = PurePosixPath(member.filename).parts
        if (
            len(parts) != 2
            or parts[0] != split
            or not COCO_IMAGE_RE.fullmatch(parts[1])
        ):
            raise zipfile.BadZipFile(f"Unexpected COCO archive member: {member.filename}")
        count += 1
    return count


def download_coco_images():
    COCO_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    for split, expected_count in COCO_IMAGE_COUNTS.items():
        image_dir = COCO_DOWNLOAD_ROOT / split
        try:
            existing_count = len(available_image_ids(image_dir))
        except FileNotFoundError:
            existing_count = 0
        if existing_count == expected_count:
            progress(f"using cached official COCO images: {split} ({existing_count:,})")
            continue

        archive_path = COCO_DOWNLOAD_ROOT / f"{split}.zip"
        if archive_path.exists() and not zipfile.is_zipfile(archive_path):
            archive_path.unlink()
        if not archive_path.is_file():
            progress(f"downloading official COCO archive: {split}.zip")
            download_resumable_https(
                f"{COCO_ARCHIVE_BASE_URL}/{split}.zip",
                archive_path,
            )

        staging_root = COCO_DOWNLOAD_ROOT / f".{split}.extracting"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                actual_count = _archive_image_count(archive, split)
                if actual_count != expected_count:
                    raise zipfile.BadZipFile(
                        f"{split}.zip contains {actual_count:,} images; "
                        f"expected {expected_count:,}"
                    )
            with progress_stage(f"extract official {split}"):
                extract_zip_safely(archive_path, staging_root)
            extracted_dir = staging_root / split
            extracted_count = len(available_image_ids(extracted_dir))
            if extracted_count != expected_count:
                raise zipfile.BadZipFile(
                    f"Extracted {split} contains {extracted_count:,} images; "
                    f"expected {expected_count:,}"
                )
            if image_dir.exists():
                shutil.rmtree(image_dir)
            extracted_dir.replace(image_dir)
            archive_path.unlink()
            progress(f"prepared official COCO images: {split} ({extracted_count:,})")
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
    return COCO_DOWNLOAD_ROOT, COCO_DOWNLOAD_ROOT / "train2014", COCO_DOWNLOAD_ROOT / "val2014"


def extract_zip_safely(archive_path, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            mode = member.external_attr >> 16
            if not parts or member.filename.startswith("/") or ".." in parts:
                raise zipfile.BadZipFile(f"Unsafe archive member: {member.filename}")
            if stat.S_ISLNK(mode):
                raise zipfile.BadZipFile(f"Unsupported archive link: {member.filename}")
            destination = target.joinpath(*parts).resolve()
            if not destination.is_relative_to(target):
                raise zipfile.BadZipFile(f"Unsafe archive member: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def download_vqa_file(filename):
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    existing = list(DOWNLOAD_ROOT.rglob(filename))
    if existing:
        progress(f"using existing annotation file: {existing[0]}")
        return existing[0]

    url = VQA_DOWNLOADS[filename]
    zip_path = DOWNLOAD_ROOT / Path(url).name
    if not zip_path.exists():
        progress(f"downloading VQA annotation archive: {Path(url).name}")
        download_https(url, zip_path)
    progress(f"extracting VQA annotation archive: {zip_path.name}")
    extract_zip_safely(zip_path, DOWNLOAD_ROOT)
    return find_file(DOWNLOAD_ROOT, filename)


def link_path(source, target):
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def restore_resume_artifacts():
    latest_checkpoint = CHECKPOINT_DIR / "latest.pt"
    if latest_checkpoint.exists():
        progress(f"using checkpoint already present at {latest_checkpoint}")
        return latest_checkpoint

    if not RESUME_ROOT.is_dir():
        progress(f"no resume dataset found at {RESUME_ROOT}; starting a new run")
        return None

    source_latest = find_file(RESUME_ROOT, "latest.pt")
    for filename in RESUME_FILES:
        try:
            source = find_file(RESUME_ROOT, filename)
        except FileNotFoundError:
            continue
        shutil.copy2(source, CHECKPOINT_DIR / filename)

    try:
        shutil.copy2(find_file(RESUME_ROOT, "answer_vocab.json"), ANSWER_VOCAB)
    except FileNotFoundError:
        pass

    restored_latest = CHECKPOINT_DIR / source_latest.name
    progress(f"restored resume checkpoint from {source_latest} to {restored_latest}")
    return restored_latest


def completed_training_epochs():
    summary_path = CHECKPOINT_DIR / "run_summary.json"
    if not summary_path.is_file():
        return 0
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(summary.get("total_epochs", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def available_image_ids(image_dir):
    image_ids = set()
    for path in image_dir.glob("*.jpg"):
        match = COCO_IMAGE_RE.match(path.name)
        if match:
            image_ids.add(int(match.group(1)))
    if not image_ids:
        raise FileNotFoundError(f"No COCO image files found in {image_dir}")
    return image_ids


def required_image_ids(questions_path):
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    return {int(item["image_id"]) for item in payload["questions"]}


def run_packaged_input_smoke():
    packaged = find_packaged_vqa_archives(KERNEL_OUTPUT_ROOT)
    if packaged is None:
        raise FileNotFoundError(
            f"No sharded COCO kernel outputs found under {KERNEL_OUTPUT_ROOT}"
        )
    data_root = normalize_vqa_data()

    for split, spec in PACKAGED_SPLITS.items():
        image_dir = data_root / split
        available = available_image_ids(image_dir)
        questions_path = data_root / spec["json_names"][0]
        required = required_image_ids(questions_path)
        missing = required - available
        if missing:
            sample = ", ".join(str(image_id) for image_id in sorted(missing)[:10])
            raise FileNotFoundError(
                f"{split} is missing {len(missing):,} question-referenced images: {sample}"
            )
        expected_count = packaged[split]["manifest"]["image_count"]
        if len(available) < expected_count:
            raise FileNotFoundError(
                f"{split} has {len(available):,} images after normalization; "
                f"package contains {expected_count:,}"
            )
        progress(
            f"validated {split}: {len(available):,} images, "
            f"{len(required):,} question-referenced images"
        )

    training_launcher(cwd=Path.cwd())
    run(
        [
            "python",
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                "count = torch.cuda.device_count(); "
                "assert count >= 2, f'Expected two T4 GPUs, found {count}'; "
                "[(torch.cuda.synchronize(i), print(i, torch.cuda.get_device_name(i), "
                "torch.ones(1, device=f'cuda:{i}').item(), flush=True)) "
                "for i in range(2)]"
            ),
        ],
        cwd=Path.cwd(),
    )
    progress(
        "sharded data, question references, and both CUDA devices passed; training was not started",
        stage="smoke complete",
    )


def run_network_smoke():
    probe_coco_archive_sources()
    training_launcher(cwd=Path.cwd())
    run(
        [
            "python",
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                "count = torch.cuda.device_count(); "
                "assert count >= 2, f'Expected two T4 GPUs, found {count}'; "
                "[(torch.cuda.synchronize(i), print(i, torch.cuda.get_device_name(i), "
                "torch.ones(1, device=f'cuda:{i}').item(), flush=True)) "
                "for i in range(2)]"
            ),
        ],
        cwd=Path.cwd(),
    )


def prepare_val_images(source_dir, target_dir, questions_path):
    if target_dir.is_symlink():
        target_dir.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for source in source_dir.glob("*.jpg"):
        link_path(source, target_dir / source.name)
        linked += 1
        if linked % 5000 == 0:
            progress(f"val2014 image links prepared: {linked:,}")

    missing = []
    for image_id in sorted(required_image_ids(questions_path)):
        filename = f"COCO_val2014_{image_id:012d}.jpg"
        if not (target_dir / filename).is_file():
            missing.append((image_id, filename))

    progress(f"val2014: repairing {len(missing):,} missing referenced images")
    for index, (_, filename) in enumerate(missing, start=1):
        target = target_dir / filename
        temporary = target.with_suffix(".jpg.part")
        url = f"{COCO_IMAGE_BASE_URL}/val2014/{filename}"
        for attempt in range(1, 4):
            try:
                download_https(url, temporary)
                temporary.replace(target)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        if index % 25 == 0 or index == len(missing):
            progress(f"val2014 repair progress: {index:,}/{len(missing):,}")
    return target_dir


def write_json(data, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    target.write_text(json.dumps(data), encoding="utf-8")


def copy_vqa_split(questions_source, annotations_source, questions_target, annotations_target):
    shutil.copy2(questions_source, questions_target)
    shutil.copy2(annotations_source, annotations_target)


def normalize_vqa_data():
    input_root = Path("/kaggle/input")
    packaged = find_packaged_vqa_archives(KERNEL_OUTPUT_ROOT)
    raw_root = extract_packaged_vqa_data(packaged) if packaged is not None else None
    normalized_root = NORMALIZED_DATA_ROOT
    if packaged is not None and NORMALIZED_DATA_ROOT == WORK_ROOT / "vqa":
        normalized_root = PACKED_DATA_ROOT / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)
    if packaged is not None:
        train_images_source = raw_root / "train2014"
        val_images_source = raw_root / "val2014"
        train_questions = find_file(raw_root, "v2_OpenEnded_mscoco_train2014_questions.json")
        train_annotations = find_file(raw_root, "v2_mscoco_train2014_annotations.json")
        val_questions = find_file(raw_root, "v2_OpenEnded_mscoco_val2014_questions.json")
        val_annotations = find_file(raw_root, "v2_mscoco_val2014_annotations.json")
        progress(
            f"using sharded COCO kernel outputs under {KERNEL_OUTPUT_ROOT}",
            stage="dataset preparation",
        )
    else:
        try:
            raw_root, train_images_source, val_images_source = resolve_vqa_data_root(
                [
                    RAW_DATA_ROOT,
                    input_root / "coco2014vqa" / "Dataset",
                    input_root / "coco2014vqa",
                    input_root / "multimodal-vqa-data" / "vqa",
                    input_root / "multimodal-vqa-data",
                    input_root / "datasets" / "sagnikkayalcse52" / "coco2014vqa" / "Dataset",
                    input_root / "datasets" / "sagnikkayalcse52" / "coco2014vqa",
                    input_root / "datasets" / "coco2014vqa" / "Dataset",
                    input_root / "datasets" / "coco2014vqa",
                    input_root / "datasets",
                ]
            )
            if not next(train_images_source.glob("COCO_train2014_*.jpg"), None):
                raise FileNotFoundError(f"No COCO train images found under {train_images_source}")
            if not next(val_images_source.glob("COCO_val2014_*.jpg"), None):
                raise FileNotFoundError(f"No COCO validation images found under {val_images_source}")
            progress(
                f"resolved mounted COCO image directories under {raw_root}",
                stage="dataset preparation",
            )
        except FileNotFoundError as exc:
            progress(
                f"no complete mounted COCO image source found ({exc}); "
                "using resumable official S3 downloads",
                stage="dataset preparation",
            )
            raw_root, train_images_source, val_images_source = download_coco_images()
        train_questions = download_vqa_file("v2_OpenEnded_mscoco_train2014_questions.json")
        train_annotations = download_vqa_file("v2_mscoco_train2014_annotations.json")
        val_questions = download_vqa_file("v2_OpenEnded_mscoco_val2014_questions.json")
        val_annotations = download_vqa_file("v2_mscoco_val2014_annotations.json")

    link_path(train_images_source, normalized_root / "train2014")
    prepare_val_images(val_images_source, normalized_root / "val2014", val_questions)

    copy_vqa_split(
        train_questions,
        train_annotations,
        normalized_root / "v2_OpenEnded_mscoco_train2014_questions.json",
        normalized_root / "v2_mscoco_train2014_annotations.json",
    )
    copy_vqa_split(
        val_questions,
        val_annotations,
        normalized_root / "v2_OpenEnded_mscoco_val2014_questions.json",
        normalized_root / "v2_mscoco_val2014_annotations.json",
    )
    progress(f"normalized VQA data root: {normalized_root}")
    return normalized_root


def run_official_evaluation(data_root):
    if not VQA_TOOLKIT_ROOT.is_dir():
        run(["git", "clone", "--depth", "1", "https://github.com/GT-Vision-Lab/VQA.git", VQA_TOOLKIT_ROOT])
    run(
        [
            "python",
            "scripts/prepare_official_vqa_toolkit.py",
            "--toolkit-root",
            VQA_TOOLKIT_ROOT,
        ],
        cwd=REPO_ROOT,
    )
    run(
        [
            "python",
            "scripts/run_official_vqa_eval.py",
            "--toolkit-root",
            VQA_TOOLKIT_ROOT,
            "--questions",
            data_root / "v2_OpenEnded_mscoco_val2014_questions.json",
            "--annotations",
            data_root / "v2_mscoco_val2014_annotations.json",
            "--predictions",
            PREDICTIONS_PATH,
            "--output",
            OFFICIAL_METRICS_PATH,
        ],
        cwd=REPO_ROOT,
    )


def archive_artifacts():
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    export_run = EXPORT_ROOT / RUN_NAME
    shutil.copytree(CHECKPOINT_DIR, export_run)
    if ANSWER_VOCAB.is_file():
        shutil.copy2(ANSWER_VOCAB, EXPORT_ROOT / "answer_vocab.json")
    archive = ARCHIVE_PATH.with_suffix(".tar.gz")
    archive.unlink(missing_ok=True)
    shutil.make_archive(str(ARCHIVE_PATH), "gztar", root_dir=EXPORT_ROOT)
    progress(f"artifacts archived at {archive}")


def main():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    initialize_progress_logging()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    progress(
        f"run={RUN_NAME}; config={CONFIG_PATH}; git_ref={GIT_REF}; "
        f"data_source={RAW_DATA_ROOT}; resume_source={RESUME_ROOT}"
    )

    with progress_stage("hardware preflight"):
        run(["nvidia-smi", "-L"])
        run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])

    if DATA_SMOKE_ONLY:
        with progress_stage("sharded data and CUDA smoke"):
            run_packaged_input_smoke()
        progress(
            "bounded data smoke completed successfully; no source checkout, dependency install, or training was run",
            state="completed",
            stage="smoke complete",
        )
        return

    if NETWORK_SMOKE_ONLY:
        with progress_stage("official COCO source and CUDA smoke"):
            run_network_smoke()
        progress(
            "official COCO sources and both CUDA devices passed; no data download or training was run",
            state="completed",
            stage="smoke complete",
        )
        return

    with progress_stage("source checkout"):
        if not REPO_ROOT.exists():
            run(["git", "clone", "https://github.com/dongtingshuo/multimodal-vqa.git", REPO_ROOT])
        run(["git", "fetch", "--all", "--tags"], cwd=REPO_ROOT)
        run(["git", "checkout", GIT_REF], cwd=REPO_ROOT)
        run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)

    with progress_stage("runtime and dependencies"):
        install_training_dependencies()
        os.environ["WANDB_MODE"] = "disabled"
        progress("W&B disabled; using local CSV/PNG/JSON artifacts only.")
        train_launcher = training_launcher()

    with progress_stage("checkpoint restore"):
        latest_checkpoint = restore_resume_artifacts()

    with progress_stage("dataset preparation"):
        data_root = normalize_vqa_data()

    with progress_stage("strict data validation"):
        run(
            [
                "python",
                "-u",
                "scripts/validate_vqa_data.py",
                "--root",
                data_root,
                "--sample-images",
                "20",
                "--strict-full",
            ],
            cwd=REPO_ROOT,
        )

    with progress_stage("distributed training"):
        train_command = train_launcher + [
            "--config",
            CONFIG_PATH,
            "--device",
            "cuda",
            "--data-root",
            data_root,
            "--answer-vocab-path",
            ANSWER_VOCAB,
            "--checkpoint-dir",
            CHECKPOINT_DIR,
            "--epochs",
            TOTAL_EPOCHS,
            "--no-wandb",
        ]
        if latest_checkpoint is not None:
            train_command.extend(["--resume", latest_checkpoint])

        completed_epochs = completed_training_epochs()
        if latest_checkpoint is not None and completed_epochs >= int(TOTAL_EPOCHS):
            progress(f"training already completed {completed_epochs}/{TOTAL_EPOCHS} epochs; skipping to evaluation")
        else:
            run(train_command, cwd=REPO_ROOT)

    with progress_stage("prediction and official evaluation"):
        run(
            [
                "python",
                "-u",
                "evaluate.py",
                "--config",
                CONFIG_PATH,
                "--checkpoint",
                CHECKPOINT_DIR / "best.pt",
                "--device",
                "cuda",
                "--data-root",
                data_root,
                "--predictions-output",
                PREDICTIONS_PATH,
            ],
            cwd=REPO_ROOT,
        )
        run_official_evaluation(data_root)

    with progress_stage("artifact packaging"):
        archive_artifacts()
    progress("Kaggle run completed successfully.", state="completed", stage="complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        progress(f"run failed: {type(exc).__name__}: {exc}", state="failed")
        raise
