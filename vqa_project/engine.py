from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

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


def forward_model(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    optional = {key: batch[key] for key in ("pixel_mask", "token_type_ids") if key in batch}
    return model(batch["images"], batch["input_ids"], batch["attention_mask"], **optional)


def vqa_bce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Return the standard per-example VQA multilabel loss."""
    batch_size = max(int(logits.shape[0]), 1)
    smoothing = max(0.0, min(float(label_smoothing), 1.0))
    if smoothing > 0:
        targets = targets.mul(1.0 - smoothing).add(0.5 * smoothing)
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
    scaler: torch.amp.GradScaler | None = None,
    gradient_accumulation_steps: int = 1,
    label_smoothing: float = 0.0,
    step_callback: Callable[[dict[str, float]], None] | None = None,
    epoch: int | None = None,
    global_step_start: int = 0,
    scheduler: Any | None = None,
) -> dict[str, float]:
    model.train()
    amp_enabled = _cuda_amp_enabled(device, use_amp)
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation_steps = max(int(gradient_accumulation_steps), 1)
    total_loss = 0.0
    total_correct = 0
    total_vqa_score = 0.0
    total_top5_vqa_score = 0.0
    total_examples = 0
    optimizer_steps = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(dataloader)
    final_group_size = total_batches % accumulation_steps
    final_group_start = total_batches - final_group_size + 1
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = forward_model(model, batch)
            loss = vqa_bce_loss(logits, batch["targets"], label_smoothing=label_smoothing)

        loss_divisor = final_group_size if final_group_size and step >= final_group_start else accumulation_steps
        scaler.scale(loss / loss_divisor).backward()
        should_step = step % accumulation_steps == 0 or step == total_batches
        if should_step:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            step_update = getattr(scheduler, "step_update", None)
            if callable(step_update):
                step_update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

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
            running = {
                "epoch": float(epoch or 0),
                "step": float(step),
                "global_step": float(global_step_start + optimizer_steps),
                "train/loss": total_loss / total_examples,
                "train/accuracy": total_correct / total_examples,
                "train/vqa_score": total_vqa_score / total_examples,
                "train/top5_vqa_score": total_top5_vqa_score / total_examples,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            running.update(
                {
                    f"train/lr/{group.get('group_name', index)}": float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
            )
            progress.set_postfix(
                loss=running["train/loss"],
                acc=running["train/accuracy"],
                vqa=running["train/vqa_score"],
                top5=running["train/top5_vqa_score"],
                lr=running["train/lr"],
            )
            if step_callback is not None:
                step_callback(running)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "vqa_score": total_vqa_score / max(total_examples, 1),
        "top5_vqa_score": total_top5_vqa_score / max(total_examples, 1),
        "optimizer_steps": float(optimizer_steps),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    answer_vocab: AnswerVocab | None = None,
    prediction_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    if prediction_records is not None and answer_vocab is None:
        raise ValueError("answer_vocab is required when collecting predictions")
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
            logits = forward_model(model, batch)
        predictions = logits.argmax(dim=1)
        labels = batch["labels"]
        known_mask = labels >= 0
        known_examples = int(known_mask.sum().item())
        if known_examples:
            known_logits = logits[known_mask]
            known_targets = batch["targets"][known_mask]
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = vqa_bce_loss(known_logits, known_targets)
            batch_scores = vqa_batch_scores(known_logits, known_targets)
            total_correct += (predictions[known_mask] == labels[known_mask]).sum().item()
            total_vqa_score += batch_scores["vqa_score"]
            total_top5_vqa_score += batch_scores["top5_vqa_score"]
            total_examples += known_examples
            total_loss += loss.item() * known_examples
        if answer_vocab is not None and prediction_records is not None:
            for question_id, prediction in zip(batch["question_ids"], predictions.tolist()):
                prediction_records.append({"question_id": int(question_id), "answer": answer_vocab.decode(prediction)})

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "vqa_score": total_vqa_score / max(total_examples, 1),
        "top5_vqa_score": total_top5_vqa_score / max(total_examples, 1),
        "evaluated_examples": float(total_examples),
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scaler: torch.amp.GradScaler | None = None,
    training_state: dict[str, Any] | None = None,
    data_generator: torch.Generator | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "idx_to_answer": answer_vocab.idx_to_answer,
        "metadata": metadata or {},
        "training_state": training_state or {},
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_generator": data_generator.get_state() if data_generator is not None else None,
        },
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def restore_training_checkpoint(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler | None,
    data_generator: torch.Generator | None,
) -> dict[str, Any]:
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" not in checkpoint:
        raise ValueError("Checkpoint does not contain optimizer state and cannot resume training.")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    rng_state = checkpoint.get("rng_state") or {}
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])
    if data_generator is not None and rng_state.get("data_generator") is not None:
        data_generator.set_state(rng_state["data_generator"].cpu())
    return dict(checkpoint.get("training_state") or {})


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)
