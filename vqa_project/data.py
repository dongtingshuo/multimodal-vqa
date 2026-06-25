from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .answers import AnswerVocab, normalize_answer


def split_to_coco_name(split: str) -> str:
    if split in {"train", "train2014"}:
        return "train2014"
    if split in {"val", "val2014"}:
        return "val2014"
    raise ValueError(f"Unsupported split: {split}")


def default_question_path(root: str | Path, split: str) -> Path:
    coco_split = split_to_coco_name(split)
    name = f"v2_OpenEnded_mscoco_{coco_split}_questions.json"
    return Path(root) / name


def default_annotation_path(root: str | Path, split: str) -> Path:
    coco_split = split_to_coco_name(split)
    name = f"v2_mscoco_{coco_split}_annotations.json"
    return Path(root) / name


def default_image_dir(root: str | Path, split: str) -> Path:
    return Path(root) / split_to_coco_name(split)


def build_image_path(image_dir: Path, image_id: int, split: str) -> Path:
    coco_split = split_to_coco_name(split)
    return image_dir / f"COCO_{coco_split}_{image_id:012d}.jpg"


def build_image_transform(image_size: int, train: bool) -> transforms.Compose:
    ops: list[Any] = []
    if train:
        ops.extend(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop((image_size, image_size)),
            ]
        )
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


class VQADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        answer_vocab: AnswerVocab,
        image_size: int = 224,
        max_samples: int | None = None,
        question_path: str | Path | None = None,
        annotation_path: str | Path | None = None,
        image_dir: str | Path | None = None,
        train: bool = False,
        filter_without_known_answer: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.answer_vocab = answer_vocab
        self.answer_to_idx = answer_vocab.answer_to_idx
        self.image_dir = Path(image_dir) if image_dir else default_image_dir(self.root, split)
        self.transform = build_image_transform(image_size, train=train)

        q_path = Path(question_path) if question_path else default_question_path(self.root, split)
        a_path = Path(annotation_path) if annotation_path else default_annotation_path(self.root, split)
        with q_path.open("r", encoding="utf-8") as f:
            questions_payload = json.load(f)
        with a_path.open("r", encoding="utf-8") as f:
            annotations_payload = json.load(f)

        questions = {item["question_id"]: item for item in questions_payload["questions"]}
        examples = []
        for annotation in annotations_payload["annotations"]:
            question = questions.get(annotation["question_id"])
            if question is None:
                continue
            target_indices, target_scores, hard_label = self._build_target(annotation)
            if filter_without_known_answer and hard_label < 0:
                continue
            examples.append(
                {
                    "question_id": annotation["question_id"],
                    "image_id": annotation["image_id"],
                    "question": question["question"],
                    "target_indices": target_indices,
                    "target_scores": target_scores,
                    "label": hard_label,
                }
            )
            if max_samples is not None and len(examples) >= max_samples:
                break

        if not examples:
            raise ValueError("No usable VQA examples were loaded. Check data paths and answer vocab coverage.")
        self.examples = examples

    def _build_target(self, annotation: dict[str, Any]) -> tuple[list[int], list[float], int]:
        answers = annotation.get("answers") or []
        if answers:
            counts: Counter[int] = Counter()
            for item in answers:
                idx = self.answer_to_idx.get(normalize_answer(item.get("answer", "")))
                if idx is not None:
                    counts[idx] += 1
            indices = list(counts)
            scores = [min(counts[idx] / 3.0, 1.0) for idx in indices]
            hard_label = counts.most_common(1)[0][0] if counts else -1
            return indices, scores, hard_label

        idx = self.answer_to_idx.get(normalize_answer(annotation.get("multiple_choice_answer", "")))
        if idx is not None:
            return [idx], [1.0], idx
        return [], [], -1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        image_path = build_image_path(self.image_dir, example["image_id"], self.split)
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return {
            "image": image_tensor,
            "question": example["question"],
            "target_indices": torch.tensor(example["target_indices"], dtype=torch.long),
            "target_scores": torch.tensor(example["target_scores"], dtype=torch.float32),
            "target_size": len(self.answer_vocab),
            "label": torch.tensor(example["label"], dtype=torch.long),
            "question_id": example["question_id"],
            "image_id": example["image_id"],
        }


class VQACollator:
    def __init__(self, tokenizer, max_question_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        questions = [item["question"] for item in batch]
        target_size = int(batch[0]["target_size"])
        targets = torch.zeros((len(batch), target_size), dtype=torch.float32)
        for row, item in enumerate(batch):
            if item["target_indices"].numel():
                targets[row, item["target_indices"]] = item["target_scores"]
        tokens = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        return {
            "images": torch.stack([item["image"] for item in batch]),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "targets": targets,
            "labels": torch.stack([item["label"] for item in batch]),
            "questions": questions,
            "question_ids": [item["question_id"] for item in batch],
            "image_ids": [item["image_id"] for item in batch],
        }
