import os
import shutil
import subprocess
from pathlib import Path

WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working/multimodal-vqa"))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/kaggle/working/multimodal-vqa-repo"))
RUN_NAME = os.environ.get("RUN_NAME", "finetune")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/kaggle_finetune.yaml")
GIT_REF = os.environ.get("GIT_REF", "e26ba28")
TOTAL_EPOCHS = os.environ.get("TOTAL_EPOCHS", "12")
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", "/kaggle/input/coco2014vqa/Dataset"))

CHECKPOINT_DIR = WORK_ROOT / RUN_NAME
ANSWER_VOCAB = WORK_ROOT / "answer_vocab.json"
PREDICTIONS_PATH = CHECKPOINT_DIR / "val_predictions.json"
ARCHIVE_PATH = Path("/kaggle/working/multimodal-vqa-finetune-artifacts")
NORMALIZED_DATA_ROOT = Path(os.environ.get("DATA_ROOT", WORK_ROOT / "vqa"))


def run(command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def first_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in candidates))


def link_path(source, target):
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


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
    image_root = first_existing([raw_root / "images", raw_root])
    annotation_root = first_existing([raw_root / "annotations", raw_root])

    mapping = {
        "train2014": first_existing([image_root / "train2014", raw_root / "train2014"]),
        "val2014": first_existing([image_root / "val2014", raw_root / "val2014"]),
        "v2_OpenEnded_mscoco_train2014_questions.json": first_existing(
            [
                annotation_root / "v2_OpenEnded_mscoco_train2014_questions.json",
                raw_root / "v2_OpenEnded_mscoco_train2014_questions.json",
            ]
        ),
        "v2_mscoco_train2014_annotations.json": first_existing(
            [
                annotation_root / "v2_mscoco_train2014_annotations.json",
                raw_root / "v2_mscoco_train2014_annotations.json",
            ]
        ),
        "v2_OpenEnded_mscoco_val2014_questions.json": first_existing(
            [
                annotation_root / "v2_OpenEnded_mscoco_val2014_questions.json",
                raw_root / "v2_OpenEnded_mscoco_val2014_questions.json",
            ]
        ),
        "v2_mscoco_val2014_annotations.json": first_existing(
            [
                annotation_root / "v2_mscoco_val2014_annotations.json",
                raw_root / "v2_mscoco_val2014_annotations.json",
            ]
        ),
    }

    NORMALIZED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, source in mapping.items():
        link_path(source, NORMALIZED_DATA_ROOT / name)
    print(f"Normalized VQA data root: {NORMALIZED_DATA_ROOT}", flush=True)
    return NORMALIZED_DATA_ROOT


def main():
    run(["nvidia-smi"])
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = normalize_vqa_data()

    if not REPO_ROOT.exists():
        run(["git", "clone", "https://github.com/dongtingshuo/multimodal-vqa.git", REPO_ROOT])

    run(["git", "fetch", "--all", "--tags"], cwd=REPO_ROOT)
    run(["git", "checkout", GIT_REF], cwd=REPO_ROOT)
    run(["python", "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO_ROOT)

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
