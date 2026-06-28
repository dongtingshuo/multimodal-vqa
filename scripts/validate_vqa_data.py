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
    parser.add_argument("--strict-full", action="store_true", help="Validate official VQA v2 counts and every image reference.")
    return parser.parse_args()


EXPECTED_COUNTS = {
    "train": {"questions": 443757, "annotations": 443757, "images": 82783},
    "val": {"questions": 214354, "annotations": 214354, "images": 40504},
}


def validate_split(root: Path, split: str, sample_images: int, strict_full: bool = False) -> dict[str, int]:
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

    image_dir = root / ("train2014" if split == "train" else "val2014")
    image_files = list(image_dir.glob("*.jpg"))
    if strict_full:
        expected = EXPECTED_COUNTS[split]
        actual = {"questions": len(questions), "annotations": len(annotations), "images": len(image_files)}
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                raise ValueError(
                    f"Incomplete {split} split: expected {expected_value} {key}, found {actual[key]}."
                )
        question_ids = {int(item["question_id"]) for item in questions}
        annotation_ids = {int(item["question_id"]) for item in annotations}
        if question_ids != annotation_ids:
            raise ValueError(f"Question/annotation ID mismatch for split '{split}'.")
        referenced_ids = {int(item["image_id"]) for item in questions}
        available_ids = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in image_files
        }
        missing_ids = referenced_ids - available_ids
        if missing_ids:
            sample = ", ".join(str(value) for value in sorted(missing_ids)[:10])
            raise FileNotFoundError(f"{split} is missing {len(missing_ids)} referenced images; sample IDs: {sample}")

    checked = 0
    for annotation in annotations:
        image_path = build_image_path(image_dir, annotation["image_id"], split)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing referenced image: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        checked += 1
        if checked >= sample_images:
            break
    return {
        "questions": len(questions),
        "annotations": len(annotations),
        "image_files": len(image_files),
        "images_checked": checked,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    results = {
        split: validate_split(root, split, args.sample_images, strict_full=args.strict_full)
        for split in ("train", "val")
    }
    print(json.dumps({"root": str(root.resolve()), "splits": results}, indent=2))


if __name__ == "__main__":
    main()
