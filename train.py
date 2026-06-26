from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vqa_project.answers import AnswerVocab, build_answer_vocab
from vqa_project.config import apply_runtime_overrides, load_config, resolve_device
from vqa_project.data import VQACollator, VQADataset, default_annotation_path
from vqa_project.engine import (
    evaluate,
    load_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
from vqa_project.hf import load_tokenizer
from vqa_project.model import build_model
from vqa_project.tracking import (
    collect_run_metadata,
    create_run_directory,
    init_wandb_tracker,
    save_training_curves,
    utc_now,
    write_json,
    write_run_summary,
    write_training_history,
)
from vqa_project.training import (
    build_optimizer,
    build_scheduler,
    configure_finetune_stage,
    count_parameters,
    stage_for_epoch,
    step_scheduler,
    validate_resume_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal VQA model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root")
    parser.add_argument("--answer-vocab-path")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--resume", help="Resume from a format-v3 latest checkpoint.")
    parser.add_argument("--run-name")
    parser.add_argument("--run-dir")
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-tags", nargs="*", help="Optional W&B tags for this run.")
    parser.set_defaults(wandb_enabled=None)
    return parser.parse_args()


def load_or_build_answer_vocab(data_cfg: dict, checkpoint: dict | None = None) -> AnswerVocab:
    if checkpoint and checkpoint.get("idx_to_answer"):
        answer_vocab = AnswerVocab(list(checkpoint["idx_to_answer"]))
        if len(answer_vocab) != int(data_cfg["answer_vocab_size"]):
            raise ValueError("Resume checkpoint answer vocabulary does not match the configured vocabulary size.")
        return answer_vocab

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
    dataset,
    batch_size: int,
    shuffle: bool,
    data_cfg: dict,
    train_cfg: dict,
    device: torch.device,
    collator,
    generator: torch.Generator | None = None,
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
        generator=generator,
    )


def _head_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        if group.get("group_name") == "head_decay":
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        device=args.device,
        data_root=args.data_root,
        answer_vocab_path=args.answer_vocab_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        run_name=args.run_name,
        run_dir=args.run_dir,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
    )
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"Using device: {device}")

    resume_checkpoint = load_checkpoint(args.resume, device) if args.resume else None
    if resume_checkpoint is not None:
        validate_resume_config(config, resume_checkpoint)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    answer_vocab = load_or_build_answer_vocab(data_cfg, resume_checkpoint)
    tokenizer = load_tokenizer(model_cfg["text_model_name"])
    collator = VQACollator(tokenizer, data_cfg["max_question_length"])

    train_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["train_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_train_samples"],
        train=True,
        augmentation=data_cfg.get("augmentation"),
    )
    val_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["val_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_val_samples"],
        train=False,
        augmentation=data_cfg.get("augmentation"),
    )
    print(f"answer_vocab={len(answer_vocab)} train_examples={len(train_dataset)} val_examples={len(val_dataset)}")

    data_generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = build_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=collator,
        generator=data_generator,
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
    optimizer = build_optimizer(model, train_cfg)
    total_epochs = int(train_cfg["epochs"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("use_amp", False) and device.type == "cuda"))

    runtime_cfg = config.get("runtime", {})
    run_name = runtime_cfg.get("run_name")
    if runtime_cfg.get("use_run_dir", False):
        artifact_dir = create_run_directory(runtime_cfg.get("run_dir", "runs"), run_name)
        config["train"]["checkpoint_dir"] = str(artifact_dir)
    else:
        artifact_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_path = artifact_dir / train_cfg["checkpoint_name"]
    latest_path = artifact_dir / train_cfg.get("latest_checkpoint_name", "latest.pt")
    history_path = artifact_dir / "training_history.csv"
    curves_path = artifact_dir / "training_curves.png"
    metadata_path = artifact_dir / "run_metadata.json"
    summary_path = artifact_dir / "run_summary.json"
    config_snapshot_path = artifact_dir / "config.snapshot.json"
    run_metadata = collect_run_metadata(args.config, device)
    run_metadata["selection_metric"] = train_cfg.get("selection_metric", "vqa_score")
    run_metadata["effective_config"] = config
    run_metadata["run_dir"] = str(artifact_dir)

    history: list[dict] = []
    best_metric = float("-inf")
    global_step = 0
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        restored = restore_training_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            data_generator,
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        history = list(restored.get("history") or [])
        best_metric = float(restored.get("best_metric", best_metric))
        global_step = int(restored.get("global_step", 0))
        epochs_without_improvement = int(restored.get("epochs_without_improvement", 0))
        run_metadata["resumed_from"] = str(Path(args.resume).resolve())
        run_metadata["resumed_epoch"] = int(resume_checkpoint["epoch"])

    if start_epoch > total_epochs:
        raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; configured epochs={total_epochs}.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_snapshot_path, config)
    resume_wandb_id = None
    if resume_checkpoint is not None:
        checkpoint_metadata = resume_checkpoint.get("metadata") or {}
        if isinstance(checkpoint_metadata, dict):
            resume_wandb_id = (checkpoint_metadata.get("wandb") or {}).get("run_id")
    tracker = init_wandb_tracker(config.get("tracking", {}), config, run_name=run_name, resume_run_id=resume_wandb_id)
    if tracker.enabled:
        run_metadata["wandb"] = {"run_id": tracker.run_id, "url": tracker.url}
    elif tracker.reason:
        run_metadata["wandb"] = {"enabled": False, "reason": tracker.reason}
    write_json(metadata_path, run_metadata)
    selection_metric = str(train_cfg.get("selection_metric", "vqa_score"))
    training_started = time.perf_counter()
    elapsed_offset = float(history[-1].get("total_seconds", 0.0)) if history else 0.0

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_started = time.perf_counter()
            stage = stage_for_epoch(model_cfg, train_cfg, epoch)
            configure_finetune_stage(model, stage, train_cfg)
            parameter_counts = count_parameters(model)
            print(
                f"epoch={epoch}/{total_epochs} stage={stage} "
                f"trainable={parameter_counts['trainable']:,}/{parameter_counts['total']:,}"
            )

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                grad_clip_norm=train_cfg["grad_clip_norm"],
                log_every=train_cfg["log_every"],
                use_amp=train_cfg.get("use_amp", False),
                scaler=scaler,
                gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
                label_smoothing=train_cfg.get("label_smoothing", 0.0),
                step_callback=lambda payload: tracker.log(payload, step=int(payload["global_step"])),
                epoch=epoch,
                global_step_start=global_step,
            )
            global_step += int(train_metrics["optimizer_steps"])
            val_metrics = evaluate(model, val_loader, device, use_amp=train_cfg.get("use_amp", False))
            if selection_metric not in val_metrics:
                available = ", ".join(sorted(val_metrics))
                raise ValueError(f"Unknown selection_metric '{selection_metric}'. Available metrics: {available}")
            selected_value = float(val_metrics[selection_metric])
            step_scheduler(scheduler, selected_value)
            current_lr = _head_learning_rate(optimizer)

            is_best = selected_value > best_metric
            if is_best:
                best_metric = selected_value
                epochs_without_improvement = 0
            elif epoch >= int(train_cfg.get("early_stopping_start_epoch", 6)):
                epochs_without_improvement += 1

            total_seconds = elapsed_offset + time.perf_counter() - training_started
            epoch_row = {
                "epoch": epoch,
                "stage": stage,
                "trainable_parameters": parameter_counts["trainable"],
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
                "total_seconds": total_seconds,
                "best_checkpoint": is_best,
            }
            history.append(epoch_row)
            training_state = {
                "history": history,
                "best_metric": best_metric,
                "global_step": global_step,
                "epochs_without_improvement": epochs_without_improvement,
                "finetune_stage": stage,
            }
            checkpoint_kwargs = {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "metrics": val_metrics,
                "config": config,
                "answer_vocab": answer_vocab,
                "metadata": run_metadata,
                "scheduler": scheduler,
                "scaler": scaler,
                "training_state": training_state,
                "data_generator": data_generator,
            }
            save_checkpoint(latest_path, **checkpoint_kwargs)
            if is_best:
                save_checkpoint(checkpoint_path, **checkpoint_kwargs)
                print(f"saved best checkpoint to {checkpoint_path}")

            write_training_history(history_path, history)
            save_training_curves(curves_path, history)
            tracker.log(
                {
                    "epoch": epoch,
                    "stage": stage,
                    "trainable_parameters": parameter_counts["trainable"],
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/vqa_score": train_metrics["vqa_score"],
                    "train/top5_vqa_score": train_metrics["top5_vqa_score"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/vqa_score": val_metrics["vqa_score"],
                    "val/top5_vqa_score": val_metrics["top5_vqa_score"],
                    "learning_rate": current_lr,
                    "best_metric": best_metric,
                    "epoch_seconds": epoch_row["epoch_seconds"],
                },
                step=global_step,
            )
            print(
                f"train_loss={train_metrics['loss']:.4f} train_vqa={train_metrics['vqa_score']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_vqa={val_metrics['vqa_score']:.4f} val_top5={val_metrics['top5_vqa_score']:.4f} "
                f"lr={current_lr:.2e} latest={latest_path}"
            )

            patience = int(train_cfg.get("early_stopping_patience", 0))
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"early stopping after {epochs_without_improvement} epochs without improvement")
                break
    finally:
        run_metadata["finished_at"] = utc_now()
        run_metadata["total_seconds"] = elapsed_offset + time.perf_counter() - training_started
        run_metadata["best_metric"] = best_metric
        run_metadata["artifacts"] = {
            "checkpoint": str(checkpoint_path),
            "latest_checkpoint": str(latest_path),
            "history": str(history_path),
            "curves": str(curves_path),
            "summary": str(summary_path),
            "config_snapshot": str(config_snapshot_path),
        }
        write_json(metadata_path, run_metadata)
        write_run_summary(summary_path, history, run_metadata)
        if tracker.enabled:
            tracker.log_artifact_file(history_path, name=f"{run_name or 'vqa'}-history", artifact_type="training-history")
            tracker.log_artifact_file(curves_path, name=f"{run_name or 'vqa'}-curves", artifact_type="training-curves")
            if config.get("tracking", {}).get("wandb", {}).get("log_checkpoints", False):
                tracker.log_artifact_file(checkpoint_path, name=f"{run_name or 'vqa'}-best", artifact_type="model")
        tracker.finish()


if __name__ == "__main__":
    main()
