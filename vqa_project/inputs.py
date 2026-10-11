from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as vision_functional

from .beit3 import beit3_tokenizer, resolve_beit3_tokenizer
from .data import ViltVQACollator, VQACollator, build_image_transform
from .hf import load_tokenizer, load_vilt_processor

DEFAULT_VILT_PROCESSOR = "dandelin/vilt-b32-finetuned-vqa"


@dataclass(frozen=True)
class InputPipeline:
    processor: Any
    collator: Any
    image_mode: str
    model_name: str
    tokenizer_path: Path | None = None


class TinyViltProcessor:
    """Deterministic ViLT-shaped processor used only by offline smoke configurations."""

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, images, text, max_length: int, **kwargs):
        _ = kwargs
        pixel_values = torch.stack(
            [
                vision_functional.pil_to_tensor(
                    vision_functional.resize(image.convert("RGB"), [self.image_size, self.image_size])
                ).float()
                / 255.0
                for image in images
            ]
        )
        token_rows = []
        for question in text:
            tokens = [101] + [sum(word.encode("utf-8")) % 3000 + 100 for word in question.split()] + [102]
            token_rows.append(tokens[:max_length])
        width = max(len(tokens) for tokens in token_rows)
        input_ids = torch.zeros(len(token_rows), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, tokens in enumerate(token_rows):
            input_ids[index, : len(tokens)] = torch.tensor(tokens)
            attention_mask[index, : len(tokens)] = 1
        return {
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones(len(images), self.image_size, self.image_size, dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
        }


def build_input_pipeline(
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    tokenizer_path: str | Path | None = None,
) -> InputPipeline:
    model_name = str(model_cfg.get("name", "cross_attention"))
    max_question_length = int(data_cfg["max_question_length"])
    if model_name == "vilt":
        processor_name = str(
            data_cfg.get("processor_name")
            or DEFAULT_VILT_PROCESSOR
        )
        processor = (
            TinyViltProcessor(int(data_cfg["image_size"]))
            if model_cfg.get("mock_backbones", False)
            else load_vilt_processor(processor_name)
        )
        return InputPipeline(
            processor=processor,
            collator=ViltVQACollator(processor, max_question_length),
            image_mode="path",
            model_name=model_name,
        )

    if model_name == "beit3_base":
        if model_cfg.get("mock_backbones", False):
            tokenizer = load_tokenizer(str(model_cfg.get("text_model_name", "distilbert-base-uncased")))
            tokenizer_path = None
        else:
            configured_path = tokenizer_path or model_cfg.get("tokenizer_path")
            tokenizer_path = (
                Path(configured_path)
                if configured_path
                else resolve_beit3_tokenizer(
                    model_cfg.get("cache_dir"),
                    local_files_only=os.environ.get("VQA_BEIT3_LOCAL_ONLY", "").lower() in {"1", "true", "yes"},
                )
            )
            tokenizer = beit3_tokenizer(tokenizer_path)
        return InputPipeline(
            processor=tokenizer,
            collator=VQACollator(tokenizer, max_question_length),
            image_mode="tensor",
            model_name=model_name,
            tokenizer_path=tokenizer_path,
        )

    tokenizer = load_tokenizer(str(model_cfg["text_model_name"]))
    return InputPipeline(
        processor=tokenizer,
        collator=VQACollator(tokenizer, max_question_length),
        image_mode="tensor",
        model_name=model_name,
    )


def encode_single_input(
    pipeline: InputPipeline,
    image: Image.Image,
    question: str,
    device: torch.device,
    image_size: int,
    max_question_length: int,
) -> dict[str, torch.Tensor]:
    if pipeline.model_name == "vilt":
        encoded = pipeline.processor(
            images=[image.convert("RGB")],
            text=[question],
            padding=True,
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt",
        )
        result = {
            "images": encoded["pixel_values"].to(device),
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        for key in ("pixel_mask", "token_type_ids"):
            if key in encoded:
                result[key] = encoded[key].to(device)
        return result

    transform = build_image_transform(
        image_size,
        train=False,
        beit3_preprocessing=pipeline.model_name == "beit3_base",
    )
    tokens = pipeline.processor(
        [question],
        padding=True,
        truncation=True,
        max_length=max_question_length,
        return_tensors="pt",
    )
    return {
        "images": transform(image.convert("RGB")).unsqueeze(0).to(device),
        "input_ids": tokens["input_ids"].to(device),
        "attention_mask": tokens["attention_mask"].to(device),
    }
