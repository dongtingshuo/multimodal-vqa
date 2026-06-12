from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .data import build_image_transform


@torch.no_grad()
def predict(
    model,
    tokenizer,
    answer_vocab,
    image_path: str | Path,
    question: str,
    device: torch.device,
    image_size: int,
    max_question_length: int,
    topk: int = 5,
) -> list[tuple[str, float]]:
    model.eval()
    transform = build_image_transform(image_size, train=False)
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    tokens = tokenizer(
        [question],
        padding=True,
        truncation=True,
        max_length=max_question_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    logits = model(image_tensor, input_ids, attention_mask)
    probabilities = torch.softmax(logits, dim=-1).squeeze(0)
    values, indices = probabilities.topk(min(topk, len(answer_vocab)))
    return [(answer_vocab.decode(idx.item()), value.item()) for value, idx in zip(values, indices)]
