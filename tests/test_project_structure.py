from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from vqa_project.answers import AnswerVocab, build_answer_vocab, normalize_answer
from vqa_project.data import VQACollator, VQADataset


def test_normalize_answer_removes_case_and_punctuation() -> None:
    assert normalize_answer("  Red, Car! ") == "red car"


def test_answer_vocab_builds_top_answers(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "answers": [
                            {"answer": "yes"},
                            {"answer": "yes"},
                            {"answer": "no"},
                        ]
                    },
                    {"multiple_choice_answer": "cat"},
                ]
            }
        ),
        encoding="utf-8",
    )
    vocab = build_answer_vocab(annotation_path, top_k=2)
    assert vocab.idx_to_answer == ["yes", "no"]


def test_answer_vocab_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocab = AnswerVocab(["yes", "no"])
    vocab.save(path)
    loaded = AnswerVocab.load(path)
    assert loaded.decode(1) == "no"
    assert loaded.encode("YES!") == 0


class TinyTokenizer:
    def __call__(self, questions, padding, truncation, max_length, return_tensors):
        import torch

        assert padding is True
        assert truncation is True
        assert return_tensors == "pt"
        width = min(max_length, 4)
        batch = len(questions)
        return {
            "input_ids": torch.ones(batch, width, dtype=torch.long),
            "attention_mask": torch.ones(batch, width, dtype=torch.long),
        }


def test_vqa_dataset_and_collator_load_tiny_sample(tmp_path: Path) -> None:
    image_dir = tmp_path / "train2014"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_dir / "COCO_train2014_000000000001.jpg")
    (tmp_path / "v2_OpenEnded_mscoco_train2014_questions.json").write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": 10,
                        "image_id": 1,
                        "question": "What color is the square?",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "v2_mscoco_train2014_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "question_id": 10,
                        "image_id": 1,
                        "answers": [{"answer": "red"} for _ in range(10)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = VQADataset(
        root=tmp_path,
        split="train",
        answer_vocab=AnswerVocab(["red", "blue"]),
        image_size=32,
        max_samples=1,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=VQACollator(TinyTokenizer(), 8))
    batch = next(iter(loader))
    assert batch["images"].shape == (1, 3, 32, 32)
    assert batch["targets"].shape == (1, 2)
    assert batch["labels"].item() == 0
    assert "target" not in dataset.examples[0]
    assert dataset.examples[0]["target_indices"] == [0]
    assert not any(isinstance(transform, transforms.RandomHorizontalFlip) for transform in dataset.transform.transforms)


def test_training_augmentation_can_be_enabled(tmp_path: Path) -> None:
    image_dir = tmp_path / "train2014"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_dir / "COCO_train2014_000000000001.jpg")
    (tmp_path / "v2_OpenEnded_mscoco_train2014_questions.json").write_text(
        json.dumps({"questions": [{"question_id": 10, "image_id": 1, "question": "What color?"}]}),
        encoding="utf-8",
    )
    (tmp_path / "v2_mscoco_train2014_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "question_id": 10,
                        "image_id": 1,
                        "answers": [{"answer": "red"} for _ in range(10)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = VQADataset(
        root=tmp_path,
        split="train",
        answer_vocab=AnswerVocab(["red"]),
        image_size=32,
        train=True,
        augmentation={"enabled": True, "random_resized_crop": True, "horizontal_flip": 0.5, "color_jitter": 0.1},
    )

    assert any(isinstance(transform, transforms.RandomResizedCrop) for transform in dataset.transform.transforms)
    assert any(isinstance(transform, transforms.RandomHorizontalFlip) for transform in dataset.transform.transforms)
