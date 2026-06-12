from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.answers import AnswerVocab, build_answer_vocab, normalize_answer


VQA_ZIPS = {
    "train_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "val_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "train_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "val_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
}

COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/{filename}"


def download(url: str, output_path: Path, retries: int = 3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"exists: {output_path}")
        return

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                total = int(response.headers.get("Content-Length", "0"))
                with temp_path.open("wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=output_path.name,
                ) as bar:
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            temp_path.replace(output_path)
            return
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}") from exc
            wait_seconds = 2 * attempt
            print(f"download failed ({attempt}/{retries}): {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        missing = [name for name in members if not (output_dir / name).exists()]
        if not missing:
            print(f"extracted: {zip_path.name}")
            return
        archive.extractall(output_dir)
    print(f"extracted: {zip_path.name}")


def coco_split(split: str) -> str:
    if split == "train":
        return "train2014"
    if split == "val":
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


def download_subset_images(root: Path, split: str, max_samples: int) -> None:
    split_name = coco_split(split)
    image_dir = root / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    image_ids = collect_image_ids(root, split, max_samples)
    print(f"{split}: {len(image_ids)} unique images for first {max_samples} VQA annotations")
    for image_id in tqdm(image_ids, desc=f"{split} images"):
        filename = image_filename(split, image_id)
        output_path = image_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        url = COCO_IMAGE_URL.format(split=split_name, filename=filename)
        download(url, output_path, retries=3)


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
            download(url, output_path, retries=3)
        return

    def fetch(job: tuple[str, Path]) -> Path:
        url, output_path = job
        download(url, output_path, retries=3)
        return output_path

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split} images"):
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small local VQA v2.0 subset.")
    parser.add_argument("--root", default="data/vqa")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--answer-vocab-path", default="data/answer_vocab.json")
    parser.add_argument("--answer-vocab-size", type=int, default=1000)
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    for name, url in VQA_ZIPS.items():
        zip_path = downloads / Path(url).name
        download(url, zip_path)
        extract_zip(zip_path, root)

    vocab_path = Path(args.answer_vocab_path)
    if vocab_path.exists():
        answer_vocab = AnswerVocab.load(vocab_path)
    else:
        answer_vocab = build_answer_vocab(annotation_path(root, "train"), args.answer_vocab_size)
        answer_vocab.save(vocab_path)

    if not args.skip_images:
        download_subset_images_parallel(
            root, "train", args.train_samples, args.image_workers, answer_vocab
        )
        download_subset_images_parallel(
            root, "val", args.val_samples, args.image_workers, answer_vocab
        )

    print("\nData preparation finished.")
    print(f"Root: {root.resolve()}")
    print("Next: python train.py --config configs/default.yaml")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
