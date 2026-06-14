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
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = _cuda_amp_enabled(device, use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(batch["images"], batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["targets"])

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["images"].size(0)
        predictions = logits.argmax(dim=1)
        labels = batch["labels"]
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size
        total_loss += loss.item() * batch_size

        if step % log_every == 0:
            progress.set_postfix(loss=total_loss / total_examples, acc=total_correct / total_examples)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = _cuda_amp_enabled(device, use_amp)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(batch["images"], batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["targets"])
        predictions = logits.argmax(dim=1)
        labels = batch["labels"]
        batch_size = batch["images"].size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size
        total_loss += loss.item() * batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    answer_vocab: AnswerVocab,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "idx_to_answer": answer_vocab.idx_to_answer,
        },
        checkpoint_path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)
