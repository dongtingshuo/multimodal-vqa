from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.data import build_image_path, default_annotation_path, default_question_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a VQA v2 / COCO 2014 data layout.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--sample-images", type=int, default=20)
    return parser.parse_args()


def validate_split(root: Path, split: str, sample_images: int) -> dict[str, int]:
    question_path = default_question_path(root, split)
    annotation_path = default_annotation_path(root, split)
    for path in (question_path, annotation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    with question_path.open(encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    with annotation_path.open(encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]
    if not questions or not annotations:
        raise ValueError(f"Empty questions or annotations for split '{split}'.")

    checked = 0
    for annotation in annotations:
        image_path = build_image_path(
            root / ("train2014" if split == "train" else "val2014"), annotation["image_id"], split
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing referenced image: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        checked += 1
        if checked >= sample_images:
            break
    return {"questions": len(questions), "annotations": len(annotations), "images_checked": checked}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    results = {split: validate_split(root, split, args.sample_images) for split in ("train", "val")}
    print(json.dumps({"root": str(root.resolve()), "splits": results}, indent=2))


if __name__ == "__main__":
    main()
