from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .answers import AnswerVocab, normalize_answer
from .vqa_normalization import normalize_vqa_answer_group, vqa_soft_targets


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


class VQAWithVisualGenomeSampler(Sampler[int]):
    """Shuffle all VQA examples and a deterministic per-epoch VG sample across ranks."""

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        seed: int,
        visual_genome_ratio: float = 0.25,
    ) -> None:
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("Sampler rank must be within a positive replica count.")
        if visual_genome_ratio < 0:
            raise ValueError("Visual Genome sample ratio cannot be negative.")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.base_count = int(dataset.vqa_example_count)
        self.visual_genome_count = int(dataset.visual_genome_example_count)
        self.visual_genome_ratio = float(visual_genome_ratio)
        self.visual_genome_sample_count = min(
            self.visual_genome_count,
            math.ceil(self.base_count * self.visual_genome_ratio),
        )
        global_count = self.base_count + self.visual_genome_sample_count
        self.num_samples = (global_count + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        vqa_indices = torch.randperm(self.base_count, generator=generator).tolist()
        vg_indices = (
            torch.randperm(self.visual_genome_count, generator=generator)[: self.visual_genome_sample_count]
            .add(self.base_count)
            .tolist()
        )
        indices = vqa_indices + vg_indices
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[index] for index in order]
        padding = self.total_size - len(indices)
        if padding:
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_image_transform(
    image_size: int,
    train: bool,
    augmentation: dict[str, Any] | None = None,
    beit3_preprocessing: bool = False,
) -> transforms.Compose:
    ops: list[Any] = []
    augmentation = augmentation or {}
    if beit3_preprocessing:
        ops.extend(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transforms.Compose(ops)
    use_augmentation = bool(train and augmentation.get("enabled", False))
    if use_augmentation and augmentation.get("random_resized_crop", True):
        ops.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))
    else:
        if train:
            ops.extend(
                [
                    transforms.Resize((image_size + 32, image_size + 32)),
                    transforms.RandomCrop((image_size, image_size)),
                ]
            )
        else:
            ops.append(transforms.Resize((image_size, image_size)))
    horizontal_flip = float(augmentation.get("horizontal_flip", 0.0))
    if use_augmentation and horizontal_flip > 0:
        ops.append(transforms.RandomHorizontalFlip(p=min(horizontal_flip, 1.0)))
    color_jitter = float(augmentation.get("color_jitter", 0.0))
    if use_augmentation and color_jitter > 0:
        jitter = min(color_jitter, 1.0)
        ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(jitter / 2, 0.5)))
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
        augmentation: dict[str, Any] | None = None,
        image_mode: str = "tensor",
        extra_qa_path: str | Path | None = None,
        beit3_preprocessing: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.answer_vocab = answer_vocab
        self.answer_to_idx = answer_vocab.answer_to_idx
        self.answer_normalization = answer_vocab.normalization
        self.image_dir = Path(image_dir) if image_dir else default_image_dir(self.root, split)
        if image_mode not in {"tensor", "path"}:
            raise ValueError("image_mode must be 'tensor' or 'path'.")
        self.image_mode = image_mode
        self.transform = build_image_transform(
            image_size,
            train=train,
            augmentation=augmentation,
            beit3_preprocessing=beit3_preprocessing,
        )

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
                    "source": "vqa_v2",
                    "answers": [str(item.get("answer", "")) for item in annotation.get("answers", [])],
                    "answer_type": annotation.get("answer_type", "unknown"),
                    "question_type": annotation.get("question_type", "unknown"),
                }
            )
            if max_samples is not None and len(examples) >= max_samples:
                break

        if not examples:
            raise ValueError("No usable VQA examples were loaded. Check data paths and answer vocab coverage.")

        self.vqa_example_count = len(examples)
        self.visual_genome_example_count = 0
        if extra_qa_path is not None:
            if not train:
                raise ValueError("Extra single-answer QA data can only be appended to a training dataset.")
            with Path(extra_qa_path).open("r", encoding="utf-8") as f:
                extra_payload = json.load(f)
            for item in extra_payload.get("qas", []):
                answer = str(item.get("answer", ""))
                answer_index = self.answer_vocab.encode(answer)
                if answer_index is None:
                    continue
                examples.append(
                    {
                        "question_id": int(item["qa_id"]),
                        "image_id": int(item["image_id"]),
                        "question": str(item["question"]),
                        "target_indices": [answer_index],
                        "target_scores": [1.0],
                        "label": answer_index,
                        "source": "visual_genome",
                        "answers": [answer],
                        "answer_type": "visual_genome",
                        "question_type": "visual_genome",
                    }
                )
            self.visual_genome_example_count = len(examples) - self.vqa_example_count
            if self.visual_genome_example_count == 0:
                raise ValueError("The Visual Genome QA manifest contains no examples in the configured answer vocabulary.")
        self.examples = examples

    def _normalize_answer(self, answer: str) -> str:
        if self.answer_normalization == "vqa_v2":
            return normalize_vqa_answer_group([answer])[0]
        return normalize_answer(answer)

    def _build_target(self, annotation: dict[str, Any]) -> tuple[list[int], list[float], int]:
        answers = annotation.get("answers") or []
        if answers:
            answer_texts = [str(item.get("answer", "")) for item in answers]
            if self.answer_normalization == "vqa_v2":
                answer_texts = normalize_vqa_answer_group(answer_texts)
                scores_by_answer = vqa_soft_targets([str(item.get("answer", "")) for item in answers])
            else:
                answer_texts = [normalize_answer(answer) for answer in answer_texts]
            counts: Counter[int] = Counter()
            for answer in answer_texts:
                idx = self.answer_to_idx.get(answer)
                if idx is not None:
                    counts[idx] += 1
            indices = list(counts)
            if self.answer_normalization == "vqa_v2":
                score_by_index = {
                    self.answer_to_idx[answer]: score
                    for answer, score in scores_by_answer.items()
                    if answer in self.answer_to_idx
                }
                scores = [score_by_index[idx] for idx in indices]
            else:
                scores = [min(counts[idx] / 3.0, 1.0) for idx in indices]
            hard_label = counts.most_common(1)[0][0] if counts else -1
            return indices, scores, hard_label

        idx = self.answer_to_idx.get(self._normalize_answer(annotation.get("multiple_choice_answer", "")))
        if idx is not None:
            return [idx], [1.0], idx
        return [], [], -1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        image_path = build_image_path(self.image_dir, example["image_id"], self.split)
        item = {
            "question": example["question"],
            "target_indices": torch.tensor(example["target_indices"], dtype=torch.long),
            "target_scores": torch.tensor(example["target_scores"], dtype=torch.float32),
            "target_size": len(self.answer_vocab),
            "label": torch.tensor(example["label"], dtype=torch.long),
            "question_id": example["question_id"],
            "image_id": example["image_id"],
            "answers": example.get("answers", []),
            "answer_type": example.get("answer_type", "unknown"),
            "question_type": example.get("question_type", "unknown"),
        }
        if self.image_mode == "path":
            item["image_path"] = str(image_path)
        else:
            with Image.open(image_path) as image:
                item["image"] = self.transform(image.convert("RGB"))
        return item


def _collate_targets(batch: list[dict[str, Any]]) -> torch.Tensor:
    target_size = int(batch[0]["target_size"])
    targets = torch.zeros((len(batch), target_size), dtype=torch.float32)
    for row, item in enumerate(batch):
        if item["target_indices"].numel():
            targets[row, item["target_indices"]] = item["target_scores"]
    return targets


class VQACollator:
    def __init__(self, tokenizer, max_question_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        questions = [item["question"] for item in batch]
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
            "targets": _collate_targets(batch),
            "labels": torch.stack([item["label"] for item in batch]),
            "questions": questions,
            "question_ids": [item["question_id"] for item in batch],
            "image_ids": [item["image_id"] for item in batch],
        }


class ViltVQACollator:
    def __init__(self, processor, max_question_length: int) -> None:
        self.processor = processor
        self.max_question_length = max_question_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        questions = [item["question"] for item in batch]
        images = []
        for item in batch:
            with Image.open(item["image_path"]) as image:
                images.append(image.convert("RGB").copy())
        encoded = self.processor(
            images=images,
            text=questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        result = {
            "images": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "targets": _collate_targets(batch),
            "labels": torch.stack([item["label"] for item in batch]),
            "questions": questions,
            "question_ids": [item["question_id"] for item in batch],
            "image_ids": [item["image_id"] for item in batch],
        }
        for key in ("pixel_mask", "token_type_ids"):
            if key in encoded:
                result[key] = encoded[key]
        return result
