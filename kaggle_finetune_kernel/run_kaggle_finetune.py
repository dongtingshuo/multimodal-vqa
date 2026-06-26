import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working/multimodal-vqa"))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/kaggle/working/multimodal-vqa-repo"))
RUN_NAME = os.environ.get("RUN_NAME", "finetune")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/kaggle_strong.yaml")
GIT_REF = os.environ.get("GIT_REF", "main")
TOTAL_EPOCHS = os.environ.get("TOTAL_EPOCHS", "24")
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", "/kaggle/input/coco2014vqa/Dataset"))
TORCH_VERSION = os.environ.get("TORCH_VERSION", "2.4.1+cu121")
TORCHVISION_VERSION = os.environ.get("TORCHVISION_VERSION", "0.19.1+cu121")
PYTORCH_INDEX_URL = os.environ.get("PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")
REQUIRE_WANDB = os.environ.get("REQUIRE_WANDB", "1") != "0"

CHECKPOINT_DIR = WORK_ROOT / RUN_NAME
ANSWER_VOCAB = WORK_ROOT / "answer_vocab.json"
PREDICTIONS_PATH = CHECKPOINT_DIR / "val_predictions.json"
ARCHIVE_PATH = Path("/kaggle/working/multimodal-vqa-finetune-artifacts")
NORMALIZED_DATA_ROOT = Path(os.environ.get("DATA_ROOT", WORK_ROOT / "vqa"))
DOWNLOAD_ROOT = WORK_ROOT / "downloads"

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


def load_kaggle_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        print("Kaggle Secrets client is not available; using existing environment variables.", flush=True)
        return

    client = UserSecretsClient()
    loaded = []
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        if os.environ.get(name):
            loaded.append(name)
            continue
        try:
            value = client.get_secret(name)
        except Exception as exc:
            print(f"Kaggle Secret {name} is not configured: {exc}", flush=True)
            continue
        if value:
            os.environ[name] = value
            loaded.append(name)
    if loaded:
        print("Loaded Kaggle Secrets: " + ", ".join(sorted(set(loaded))), flush=True)
    else:
        print("No W&B Kaggle Secrets were loaded; training will run without W&B.", flush=True)
    if REQUIRE_WANDB and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is required for this Kaggle run but was not available. "
            "Add a Kaggle Secret named WANDB_API_KEY and make sure it is attached/enabled "
            "for the multimodal-vqa-finetune notebook before starting the run. "
            "Set REQUIRE_WANDB=0 only when intentionally running without W&B."
        )


def install_training_dependencies():
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


def available_image_ids(image_dir):
    image_ids = set()
    for path in image_dir.glob("*.jpg"):
        match = COCO_IMAGE_RE.match(path.name)
        if match:
            image_ids.add(int(match.group(1)))
    if not image_ids:
        raise FileNotFoundError(f"No COCO image files found in {image_dir}")
    return image_ids


def write_json(data, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    target.write_text(json.dumps(data), encoding="utf-8")


def filter_vqa_split(split_name, image_dir, questions_source, annotations_source, questions_target, annotations_target):
    image_ids = available_image_ids(image_dir)
    questions = json.loads(questions_source.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_source.read_text(encoding="utf-8"))

    original_questions = len(questions["questions"])
    original_annotations = len(annotations["annotations"])
    questions["questions"] = [item for item in questions["questions"] if int(item["image_id"]) in image_ids]
    annotations["annotations"] = [item for item in annotations["annotations"] if int(item["image_id"]) in image_ids]

    if not questions["questions"] or not annotations["annotations"]:
        raise RuntimeError(f"{split_name} has no usable VQA samples after filtering missing images")

    write_json(questions, questions_target)
    write_json(annotations, annotations_target)
    print(
        f"{split_name}: kept {len(questions['questions'])}/{original_questions} questions and "
        f"{len(annotations['annotations'])}/{original_annotations} annotations with available images",
        flush=True,
    )


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
    train_images = find_dir(raw_root, "train2014")
    val_images = find_dir(raw_root, "val2014")
    link_path(train_images, NORMALIZED_DATA_ROOT / "train2014")
    link_path(val_images, NORMALIZED_DATA_ROOT / "val2014")

    filter_vqa_split(
        "train",
        train_images,
        download_vqa_file("v2_OpenEnded_mscoco_train2014_questions.json"),
        download_vqa_file("v2_mscoco_train2014_annotations.json"),
        NORMALIZED_DATA_ROOT / "v2_OpenEnded_mscoco_train2014_questions.json",
        NORMALIZED_DATA_ROOT / "v2_mscoco_train2014_annotations.json",
    )
    filter_vqa_split(
        "val",
        val_images,
        download_vqa_file("v2_OpenEnded_mscoco_val2014_questions.json"),
        download_vqa_file("v2_mscoco_val2014_annotations.json"),
        NORMALIZED_DATA_ROOT / "v2_OpenEnded_mscoco_val2014_questions.json",
        NORMALIZED_DATA_ROOT / "v2_mscoco_val2014_annotations.json",
    )
    print(f"Normalized VQA data root: {NORMALIZED_DATA_ROOT}", flush=True)
    return NORMALIZED_DATA_ROOT


def main():
    run(["nvidia-smi"])
    load_kaggle_secrets()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = normalize_vqa_data()

    if not REPO_ROOT.exists():
        run(["git", "clone", "https://github.com/dongtingshuo/multimodal-vqa.git", REPO_ROOT])

    run(["git", "fetch", "--all", "--tags"], cwd=REPO_ROOT)
    run(["git", "checkout", GIT_REF], cwd=REPO_ROOT)
    install_training_dependencies()

    run(
        [
            "python",
            "scripts/validate_vqa_data.py",
            "--root",
            data_root,
            "--sample-images",
            "20",
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
    ]
    if os.environ.get("WANDB_API_KEY"):
        train_command.extend(["--wandb", "--wandb-tags", "kaggle", "strong-cross-attention"])

    latest_checkpoint = CHECKPOINT_DIR / "latest.pt"
    if latest_checkpoint.exists():
        train_command.extend(["--resume", latest_checkpoint])

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

    if ARCHIVE_PATH.with_suffix(".tar.gz").exists():
        ARCHIVE_PATH.with_suffix(".tar.gz").unlink()
    shutil.make_archive(str(ARCHIVE_PATH), "gztar", root_dir=WORK_ROOT)
    print(f"Artifacts archived at {ARCHIVE_PATH}.tar.gz", flush=True)


if __name__ == "__main__":
    main()
