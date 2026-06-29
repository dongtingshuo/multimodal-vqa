import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working/multimodal-vqa"))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/kaggle/working/multimodal-vqa-repo"))
RUN_NAME = os.environ.get("RUN_NAME", "vilt-seed42")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/kaggle_vilt.yaml")
GIT_REF = os.environ.get("GIT_REF", "main")
TOTAL_EPOCHS = os.environ.get("TOTAL_EPOCHS", "10")
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", "/kaggle/input/coco2014vqa/Dataset"))
RESUME_ROOT = Path(
    os.environ.get("RESUME_ROOT", "/kaggle/input/multimodal-vqa-vilt-resume")
)
TORCH_VERSION = os.environ.get("TORCH_VERSION", "2.4.1+cu121")
TORCHVISION_VERSION = os.environ.get("TORCHVISION_VERSION", "0.19.1+cu121")
PYTORCH_INDEX_URL = os.environ.get("PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")
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


def run(command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def preinstalled_torch_is_usable():
    if os.environ.get("FORCE_TORCH_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
        print("FORCE_TORCH_INSTALL is enabled; installing the pinned PyTorch stack", flush=True)
        return False

    probe = subprocess.run(
        [
            "python",
            "-c",
            (
                "import torch, torchvision; "
                "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                "print(f'Using preinstalled torch={torch.__version__} '"
                "f'torchvision={torchvision.__version__} cuda={torch.version.cuda}')"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if probe.stdout.strip():
        print(probe.stdout.strip(), flush=True)
    if probe.returncode == 0:
        return True
    if probe.stderr.strip():
        print(f"Preinstalled PyTorch probe failed: {probe.stderr.strip()}", flush=True)
    return False


def install_training_dependencies():
    if not preinstalled_torch_is_usable():
        run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--index-url",
                PYTORCH_INDEX_URL,
                f"torch=={TORCH_VERSION}",
                f"torchvision=={TORCHVISION_VERSION}",
            ],
            cwd=REPO_ROOT,
        )
    run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "transformers>=4.40",
            "Pillow>=10.0",
            "PyYAML>=6.0",
            "tqdm>=4.66",
            "matplotlib>=3.8",
            "wandb>=0.17",
        ],
        cwd=REPO_ROOT,
    )


def first_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in candidates))


def find_dir(root, name):
    direct = root / name
    if direct.is_dir():
        return direct
    for path in root.rglob(name):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find directory {name!r} under {root}")


def find_file(root, name):
    direct = root / name
    if direct.is_file():
        return direct
    for path in root.rglob(name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find file {name!r} under {root}")


def download_vqa_file(filename):
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    existing = list(DOWNLOAD_ROOT.rglob(filename))
    if existing:
        return existing[0]

    url = VQA_DOWNLOADS[filename]
    zip_path = DOWNLOAD_ROOT / Path(url).name
    if not zip_path.exists():
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DOWNLOAD_ROOT)
    return find_file(DOWNLOAD_ROOT, filename)


def link_path(source, target):
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def restore_resume_artifacts():
    latest_checkpoint = CHECKPOINT_DIR / "latest.pt"
    if latest_checkpoint.exists():
        print(f"Using checkpoint already present at {latest_checkpoint}", flush=True)
        return latest_checkpoint

    if not RESUME_ROOT.is_dir():
        print(f"No resume dataset found at {RESUME_ROOT}; starting a new run", flush=True)
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
    print(f"Restored resume checkpoint from {source_latest} to {restored_latest}", flush=True)
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


def prepare_val_images(source_dir, target_dir, questions_path):
    if target_dir.is_symlink():
        target_dir.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.jpg"):
        link_path(source, target_dir / source.name)

    missing = []
    for image_id in sorted(required_image_ids(questions_path)):
        filename = f"COCO_val2014_{image_id:012d}.jpg"
        if not (target_dir / filename).is_file():
            missing.append((image_id, filename))

    print(f"val2014: repairing {len(missing)} missing referenced images", flush=True)
    for index, (_, filename) in enumerate(missing, start=1):
        target = target_dir / filename
        temporary = target.with_suffix(".jpg.part")
        url = f"{COCO_IMAGE_BASE_URL}/val2014/{filename}"
        for attempt in range(1, 4):
            try:
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(target)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        if index % 25 == 0 or index == len(missing):
            print(f"val2014 repair progress: {index}/{len(missing)}", flush=True)
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
    raw_root = first_existing(
        [
            RAW_DATA_ROOT,
            input_root / "coco2014vqa" / "Dataset",
            input_root / "coco2014vqa",
            input_root / "multimodal-vqa-data" / "vqa",
            input_root / "multimodal-vqa-data",
        ]
    )

    NORMALIZED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train_images_source = find_dir(raw_root, "train2014")
    val_images_source = find_dir(raw_root, "val2014")
    train_questions = download_vqa_file("v2_OpenEnded_mscoco_train2014_questions.json")
    train_annotations = download_vqa_file("v2_mscoco_train2014_annotations.json")
    val_questions = download_vqa_file("v2_OpenEnded_mscoco_val2014_questions.json")
    val_annotations = download_vqa_file("v2_mscoco_val2014_annotations.json")

    link_path(train_images_source, NORMALIZED_DATA_ROOT / "train2014")
    prepare_val_images(val_images_source, NORMALIZED_DATA_ROOT / "val2014", val_questions)

    copy_vqa_split(
        train_questions,
        train_annotations,
        NORMALIZED_DATA_ROOT / "v2_OpenEnded_mscoco_train2014_questions.json",
        NORMALIZED_DATA_ROOT / "v2_mscoco_train2014_annotations.json",
    )
    copy_vqa_split(
        val_questions,
        val_annotations,
        NORMALIZED_DATA_ROOT / "v2_OpenEnded_mscoco_val2014_questions.json",
        NORMALIZED_DATA_ROOT / "v2_mscoco_val2014_annotations.json",
    )
    print(f"Normalized VQA data root: {NORMALIZED_DATA_ROOT}", flush=True)
    return NORMALIZED_DATA_ROOT


def load_wandb_api_key():
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        from kaggle_secrets import UserSecretsClient

        last_error = None
        for attempt in range(1, 6):
            try:
                api_key = UserSecretsClient().get_secret("WANDB_API_KEY").strip()
                if api_key:
                    break
            except Exception as error:
                last_error = error
                print(f"W&B Secret read attempt {attempt}/5 failed: {type(error).__name__}", flush=True)
            time.sleep(min(5 * attempt, 20))
        if not api_key:
            raise RuntimeError("WANDB_API_KEY could not be read from Kaggle Secrets") from last_error
    return api_key


def configure_wandb(api_key):
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = "online"
    os.environ.setdefault("WANDB_PROJECT", "multimodal-vqa")
    import wandb

    if not wandb.login(key=api_key, relogin=True, verify=True):
        raise RuntimeError("W&B authentication failed")
    print(f"W&B authenticated; project={os.environ['WANDB_PROJECT']} mode=online", flush=True)


def run_official_evaluation(data_root):
    if not VQA_TOOLKIT_ROOT.is_dir():
        run(["git", "clone", "--depth", "1", "https://github.com/GT-Vision-Lab/VQA.git", VQA_TOOLKIT_ROOT])
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
    print(f"Artifacts archived at {archive}", flush=True)


def main():
    run(["nvidia-smi"])
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_ROOT.exists():
        run(["git", "clone", "https://github.com/dongtingshuo/multimodal-vqa.git", REPO_ROOT])

    run(["git", "fetch", "--all", "--tags"], cwd=REPO_ROOT)
    run(["git", "checkout", GIT_REF], cwd=REPO_ROOT)
    wandb_api_key = load_wandb_api_key()
    install_training_dependencies()
    configure_wandb(wandb_api_key)
    latest_checkpoint = restore_resume_artifacts()
    data_root = normalize_vqa_data()

    run(
        [
            "python",
            "scripts/validate_vqa_data.py",
            "--root",
            data_root,
            "--sample-images",
            "20",
            "--strict-full",
        ],
        cwd=REPO_ROOT,
    )
    train_command = [
        "python",
        "train.py",
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
        "--wandb",
        "--wandb-project",
        "multimodal-vqa",
        "--wandb-tags",
        "kaggle",
        "vilt",
        "coco2014-vqa",
    ]
    if latest_checkpoint is not None:
        train_command.extend(["--resume", latest_checkpoint])

    completed_epochs = completed_training_epochs()
    if latest_checkpoint is not None and completed_epochs >= int(TOTAL_EPOCHS):
        print(
            f"Training already completed {completed_epochs}/{TOTAL_EPOCHS} epochs; skipping to evaluation",
            flush=True,
        )
    else:
        run(train_command, cwd=REPO_ROOT)

    run(
        [
            "python",
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

    archive_artifacts()


if __name__ == "__main__":
    main()
