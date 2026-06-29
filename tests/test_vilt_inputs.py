from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

import vqa_project.inputs as inputs_module
from vqa_project.data import ViltVQACollator
from vqa_project.inputs import DEFAULT_VILT_PROCESSOR, TinyViltProcessor, build_input_pipeline


class FakeViltProcessor:
    def __call__(self, images, text, **kwargs):
        _ = kwargs
        assert len(images) == len(text)
        return {
            "pixel_values": torch.ones(len(images), 3, 32, 32),
            "pixel_mask": torch.ones(len(images), 32, 32, dtype=torch.long),
            "input_ids": torch.ones(len(images), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(images), 4, dtype=torch.long),
            "token_type_ids": torch.zeros(len(images), 4, dtype=torch.long),
        }


def test_vilt_collator_processes_image_paths_and_soft_targets(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (40, 40), color="white").save(image_path)
    item = {
        "image_path": str(image_path),
        "question": "What is shown?",
        "target_indices": torch.tensor([1, 3]),
        "target_scores": torch.tensor([1.0, 0.3]),
        "target_size": 5,
        "label": torch.tensor(1),
        "question_id": 7,
        "image_id": 9,
    }
    batch = ViltVQACollator(FakeViltProcessor(), max_question_length=8)([item])
    assert batch["images"].shape == (1, 3, 32, 32)
    assert batch["targets"][0].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.3, 0.0])
    assert "pixel_mask" in batch
    assert "token_type_ids" in batch


def test_mock_vilt_pipeline_never_requires_a_hugging_face_download() -> None:
    pipeline = build_input_pipeline(
        {"name": "vilt", "mock_backbones": True},
        {"image_size": 32, "max_question_length": 8},
    )
    assert isinstance(pipeline.processor, TinyViltProcessor)
    assert pipeline.image_mode == "path"


def test_vilt_pipeline_uses_processor_repository_with_complete_assets(monkeypatch) -> None:
    requested = []
    processor = FakeViltProcessor()
    monkeypatch.setattr(
        inputs_module,
        "load_vilt_processor",
        lambda model_name: requested.append(model_name) or processor,
    )

    pipeline = build_input_pipeline(
        {"name": "vilt", "pretrained_model_name": "dandelin/vilt-b32-mlm-itm"},
        {"image_size": 384, "max_question_length": 40},
    )

    assert requested == [DEFAULT_VILT_PROCESSOR]
    assert pipeline.processor is processor
