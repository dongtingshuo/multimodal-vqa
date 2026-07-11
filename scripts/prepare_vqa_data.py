from __future__ import annotations

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.answers import AnswerVocab, build_answer_vocab, normalize_answer
from vqa_project.downloads import download_file

VQA_ZIPS = {
    "train_annotations": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "checksum": "sha256:fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853",
        "size": 21708861,
    },
    "val_annotations": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "checksum": "sha256:0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e",
        "size": 10518930,
    },
    "train_questions": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "checksum": "sha256:05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e",
        "size": 7239401,
    },
    "val_questions": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "checksum": "sha256:e71f6c5c3e97a51d050f28243e262b28cd0c48d11a6b4632d769d30d3f93222a",
        "size": 3494929,
    },
}

COCO_ZIPS = {
    "train2014": {
        "url": "https://images.cocodataset.org/zips/train2014.zip",
        "checksum": "md5:0da8c0bd3d6becc4dcb32757491aca88",
        "size": 13510573713,
    },
    "val2014": {
        "url": "https://images.cocodataset.org/zips/val2014.zip",
        "checksum": "md5:a3d79f5ed8d289b7a7554ce06a5782b3",
        "size": 6645013297,
    },
}

EXPECTED_COCO_IMAGE_COUNTS = {
    "train2014": 82783,
    "val2014": 40504,
}

COCO_IMAGE_URL = "https://images.cocodataset.org/{split}/{filename}"


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        output_root = output_dir.resolve()
        for member in members:
            target = (output_dir / member).resolve()
            if not target.is_relative_to(output_root):
                raise zipfile.BadZipFile(f"Unsafe archive member: {member}")
        missing = [name for name in members if not (output_dir / name).exists()]
        if not missing:
            print(f"extracted: {zip_path.name}")
            return
        archive.extractall(output_dir)
    print(f"extracted: {zip_path.name}")


def coco_split(split: str) -> str:
    if split in {"train", "train2014"}:
        return "train2014"
    if split in {"val", "val2014"}:
        return "val2014"
    raise ValueError(split)


def annotation_path(root: Path, split: str) -> Path:
    return root / f"v2_mscoco_{coco_split(split)}_annotations.json"


def image_filename(split: str, image_id: int) -> str:
    return f"COCO_{coco_split(split)}_{image_id:012d}.jpg"


def annotation_has_known_answer(annotation: dict, answer_to_idx: dict[str, int]) -> bool:
    answers = annotation.get("answers") or []
    if answers:
        return any(normalize_answer(item.get("answer", "")) in answer_to_idx for item in answers)
    return normalize_answer(annotation.get("multiple_choice_answer", "")) in answer_to_idx


def collect_image_ids(
    root: Path,
    split: str,
    max_samples: int,
    answer_vocab: AnswerVocab | None = None,
) -> list[int]:
    with annotation_path(root, split).open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    image_ids: list[int] = []
    seen: set[int] = set()
    usable_examples = 0
    answer_to_idx = answer_vocab.answer_to_idx if answer_vocab else None
    for annotation in annotations:
        if answer_to_idx is not None and not annotation_has_known_answer(annotation, answer_to_idx):
            continue
        image_id = int(annotation["image_id"])
        if image_id not in seen:
            seen.add(image_id)
            image_ids.append(image_id)
        usable_examples += 1
        if usable_examples >= max_samples:
            break
    return image_ids


def download_subset_images_parallel(
    root: Path, split: str, max_samples: int, workers: int, answer_vocab: AnswerVocab
) -> None:
    split_name = coco_split(split)
    image_dir = root / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    image_ids = collect_image_ids(root, split, max_samples, answer_vocab=answer_vocab)
    print(f"{split}: {len(image_ids)} unique images for first {max_samples} usable VQA examples")

    jobs = []
    for image_id in image_ids:
        filename = image_filename(split, image_id)
        output_path = image_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        url = COCO_IMAGE_URL.format(split=split_name, filename=filename)
        jobs.append((url, output_path))

    if not jobs:
        print(f"{split}: all images already exist")
        return

    if workers <= 1:
        for url, output_path in tqdm(jobs, desc=f"{split} images"):
            download_file(url, output_path, retries=3)
        return

    def fetch(job: tuple[str, Path]) -> Path:
        url, output_path = job
        download_file(url, output_path, retries=3)
        return output_path

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split} images"):
            future.result()


def download_full_coco_images(root: Path, downloads: Path) -> None:
    for split_name, artifact in COCO_ZIPS.items():
        image_dir = root / split_name
        zip_path = downloads / Path(artifact["url"]).name
        existing_count = sum(1 for _ in image_dir.glob("*.jpg")) if image_dir.exists() else 0
        expected_count = EXPECTED_COCO_IMAGE_COUNTS[split_name]
        if existing_count >= expected_count:
            print(f"{split_name}: found {existing_count} images; skipping zip download")
            continue
        if existing_count > 0:
            print(f"{split_name}: found only {existing_count}/{expected_count} images; downloading full zip")
        download_file(
            artifact["url"],
            zip_path,
            expected_checksum=artifact["checksum"],
            expected_size=artifact["size"],
            retries=3,
        )
        extract_zip(zip_path, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VQA v2.0 data for local or full training.")
    parser.add_argument("--root", default="data/vqa")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--answer-vocab-path", default="data/answer_vocab.json")
    parser.add_argument("--answer-vocab-size", type=int, default=3000)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument(
        "--full-coco-images",
        action="store_true",
        help="Download and extract full COCO train2014/val2014 image zip files. Required for full VQA training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    for artifact in VQA_ZIPS.values():
        zip_path = downloads / Path(artifact["url"]).name
        download_file(
            artifact["url"],
            zip_path,
            expected_checksum=artifact["checksum"],
            expected_size=artifact["size"],
        )
        extract_zip(zip_path, root)

    vocab_path = Path(args.answer_vocab_path)
    if vocab_path.exists():
        answer_vocab = AnswerVocab.load(vocab_path)
        if len(answer_vocab) != args.answer_vocab_size:
            print(f"answer vocab size changed from {len(answer_vocab)} to {args.answer_vocab_size}; rebuilding")
            answer_vocab = build_answer_vocab(annotation_path(root, "train"), args.answer_vocab_size)
            answer_vocab.save(vocab_path)
    else:
        answer_vocab = build_answer_vocab(annotation_path(root, "train"), args.answer_vocab_size)
        answer_vocab.save(vocab_path)

    if not args.skip_images:
        if args.full_coco_images:
            download_full_coco_images(root, downloads)
        else:
            download_subset_images_parallel(root, "train", args.train_samples, args.image_workers, answer_vocab)
            download_subset_images_parallel(root, "val", args.val_samples, args.image_workers, answer_vocab)

    print()
    print("Data preparation finished.")
    print(f"Root: {root.resolve()}")
    if args.full_coco_images:
        print("Next: python train.py --config configs/default.yaml")
    else:
        print("Subset images prepared. Use --full-coco-images before full training.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
