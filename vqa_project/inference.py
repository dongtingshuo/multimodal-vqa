from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .engine import forward_model
from .inputs import InputPipeline, encode_single_input


@torch.no_grad()
def predict(
    model,
    input_pipeline: InputPipeline,
    answer_vocab,
    image_path: str | Path,
    question: str,
    device: torch.device,
    image_size: int,
    max_question_length: int,
    topk: int = 5,
) -> list[tuple[str, float]]:
    model.eval()
    with Image.open(image_path) as image:
        batch = encode_single_input(
            input_pipeline,
            image,
            question,
            device,
            image_size,
            max_question_length,
        )
    logits = forward_model(model, batch)
    probabilities = torch.softmax(logits, dim=-1).squeeze(0)
    values, indices = probabilities.topk(min(topk, len(answer_vocab)))
    return [(answer_vocab.decode(idx.item()), value.item()) for value, idx in zip(values, indices)]
