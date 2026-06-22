from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .answers import AnswerVocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


def _cuda_amp_enabled(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


def vqa_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the standard per-example VQA multilabel loss."""
    batch_size = max(int(logits.shape[0]), 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / batch_size


def vqa_batch_scores(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=1)
    vqa_score = targets.gather(1, predictions.unsqueeze(1)).sum().item()
    topk = min(5, int(logits.shape[1]))
    topk_predictions = logits.topk(topk, dim=1).indices
    top5_vqa_score = targets.gather(1, topk_predictions).max(dim=1).values.sum().item()
    return {"vqa_score": vqa_score, "top5_vqa_score": top5_vqa_score}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    log_every: int = 20,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    amp_enabled = _cuda_amp_enabled(device, use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_loss = 0.0
    total_correct = 0
    total_vqa_score = 0.0
    total_top5_vqa_score = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch["images"], batch["input_ids"], batch["attention_mask"])
            loss = vqa_bce_loss(logits, batch["targets"])

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["images"].size(0)
        predictions = logits.argmax(dim=1)
        labels = batch["labels"]
        batch_scores = vqa_batch_scores(logits, batch["targets"])
        total_correct += (predictions == labels).sum().item()
        total_vqa_score += batch_scores["vqa_score"]
        total_top5_vqa_score += batch_scores["top5_vqa_score"]
        total_examples += batch_size
        total_loss += loss.item() * batch_size

        if step % max(log_every, 1) == 0:
            progress.set_postfix(
                loss=total_loss / total_examples,
                acc=total_correct / total_examples,
                vqa=total_vqa_score / total_examples,
                lr=optimizer.param_groups[0]["lr"],
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "vqa_score": total_vqa_score / max(total_examples, 1),
        "top5_vqa_score": total_top5_vqa_score / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    amp_enabled = _cuda_amp_enabled(device, use_amp)
    total_loss = 0.0
    total_correct = 0
    total_vqa_score = 0.0
    total_top5_vqa_score = 0.0
    total_examples = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch["images"], batch["input_ids"], batch["attention_mask"])
            loss = vqa_bce_loss(logits, batch["targets"])
        predictions = logits.argmax(dim=1)
        labels = batch["labels"]
        batch_scores = vqa_batch_scores(logits, batch["targets"])
        batch_size = batch["images"].size(0)
        total_correct += (predictions == labels).sum().item()
        total_vqa_score += batch_scores["vqa_score"]
        total_top5_vqa_score += batch_scores["top5_vqa_score"]
        total_examples += batch_size
        total_loss += loss.item() * batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "vqa_score": total_vqa_score / max(total_examples, 1),
        "top5_vqa_score": total_top5_vqa_score / max(total_examples, 1),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    answer_vocab: AnswerVocab,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "idx_to_answer": answer_vocab.idx_to_answer,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)
