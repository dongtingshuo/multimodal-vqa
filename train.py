from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vqa_project.answers import AnswerVocab, build_answer_vocab
from vqa_project.config import load_config, resolve_device
from vqa_project.data import VQACollator, VQADataset, default_annotation_path
from vqa_project.engine import evaluate, save_checkpoint, set_seed, train_one_epoch
from vqa_project.hf import load_tokenizer
from vqa_project.model import build_model
from vqa_project.tracking import (
    collect_run_metadata,
    save_training_curves,
    utc_now,
    write_json,
    write_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet + DistilBERT VQA model.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def load_or_build_answer_vocab(data_cfg: dict) -> AnswerVocab:
    vocab_path = Path(data_cfg["answer_vocab_path"])
    expected_size = int(data_cfg["answer_vocab_size"])
    if vocab_path.exists():
        answer_vocab = AnswerVocab.load(vocab_path)
        if len(answer_vocab) == expected_size:
            return answer_vocab
        print(f"answer vocab size changed from {len(answer_vocab)} to {expected_size}; rebuilding {vocab_path}")

    train_annotations = default_annotation_path(data_cfg["root"], data_cfg["train_split"])
    answer_vocab = build_answer_vocab(train_annotations, expected_size)
    answer_vocab.save(vocab_path)
    return answer_vocab


def build_dataloader(
    dataset, batch_size: int, shuffle: bool, data_cfg: dict, train_cfg: dict, device: torch.device, collator
):
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"Using device: {device}")

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    answer_vocab = load_or_build_answer_vocab(data_cfg)
    tokenizer = load_tokenizer(model_cfg["text_model_name"])
    collator = VQACollator(tokenizer, data_cfg["max_question_length"])

    train_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["train_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_train_samples"],
        train=True,
    )
    val_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["val_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_val_samples"],
        train=False,
    )
    print(f"answer_vocab={len(answer_vocab)} train_examples={len(train_dataset)} val_examples={len(val_dataset)}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=collator,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=collator,
    )

    model = build_model(model_cfg, answer_vocab_size=len(answer_vocab)).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = None
    if train_cfg.get("use_scheduler", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 1)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )

    checkpoint_path = Path(train_cfg["checkpoint_dir"]) / train_cfg["checkpoint_name"]
    artifact_dir = checkpoint_path.parent
    history_path = artifact_dir / "training_history.csv"
    curves_path = artifact_dir / "training_curves.png"
    metadata_path = artifact_dir / "run_metadata.json"
    run_metadata = collect_run_metadata(args.config, device)
    run_metadata["selection_metric"] = train_cfg.get("selection_metric", "vqa_score")
    write_json(metadata_path, run_metadata)

    history: list[dict] = []
    best_metric = float("-inf")
    selection_metric = str(train_cfg.get("selection_metric", "vqa_score"))
    training_started = time.perf_counter()
    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=train_cfg["grad_clip_norm"],
            log_every=train_cfg["log_every"],
            use_amp=train_cfg.get("use_amp", False),
        )
        val_metrics = evaluate(model, val_loader, device, use_amp=train_cfg.get("use_amp", False))
        current_lr = float(optimizer.param_groups[0]["lr"])
        if selection_metric not in val_metrics:
            available = ", ".join(sorted(val_metrics))
            raise ValueError(f"Unknown selection_metric '{selection_metric}'. Available metrics: {available}")
        selected_value = float(val_metrics[selection_metric])
        if scheduler is not None:
            scheduler.step(selected_value)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_vqa={train_metrics['vqa_score']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_vqa={val_metrics['vqa_score']:.4f} "
            f"val_top5_vqa={val_metrics['top5_vqa_score']:.4f} "
            f"lr={current_lr:.2e}"
        )

        is_best = selected_value > best_metric
        if is_best:
            best_metric = selected_value
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
                answer_vocab,
                metadata=run_metadata,
            )
            print(f"saved best checkpoint to {checkpoint_path}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_vqa_score": train_metrics["vqa_score"],
                "train_top5_vqa_score": train_metrics["top5_vqa_score"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_vqa_score": val_metrics["vqa_score"],
                "val_top5_vqa_score": val_metrics["top5_vqa_score"],
                "learning_rate": current_lr,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "total_seconds": time.perf_counter() - training_started,
                "best_checkpoint": is_best,
            }
        )
        write_training_history(history_path, history)
        save_training_curves(curves_path, history)

    run_metadata["finished_at"] = utc_now()
    run_metadata["total_seconds"] = time.perf_counter() - training_started
    run_metadata["best_metric"] = best_metric
    run_metadata["artifacts"] = {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "curves": str(curves_path),
    }
    write_json(metadata_path, run_metadata)


if __name__ == "__main__":
    main()
