from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from vqa_project.answers import AnswerVocab, build_answer_vocab
from vqa_project.config import apply_runtime_overrides, load_config, resolve_device
from vqa_project.data import VQADataset, default_annotation_path
from vqa_project.engine import (
    capture_rng_state,
    evaluate,
    load_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
from vqa_project.inputs import build_input_pipeline
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
    sampler=None,
):
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
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


def _learning_rate_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        f"learning_rate/{group.get('group_name', index)}": float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def initialize_distributed(configured_device: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        return resolve_device(configured_device), rank, world_size, local_rank
    if configured_device not in {"auto", "cuda"}:
        raise ValueError("Distributed training requires CUDA; set device to 'auto' or 'cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun requested multiple workers, but CUDA is unavailable.")
    if not distributed.is_nccl_available():
        raise RuntimeError("Multi-GPU CUDA training requires a PyTorch build with NCCL support (Linux CUDA runtime).")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"torchrun local rank {local_rank} exceeds the visible CUDA device count {torch.cuda.device_count()}."
        )
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(hours=2),
    )
    return torch.device("cuda", local_rank), rank, world_size, local_rank


def effective_batch_plan(batch_size: int, accumulation_steps: int, world_size: int, preserve: bool = True):
    batch_size = max(int(batch_size), 1)
    accumulation_steps = max(int(accumulation_steps), 1)
    world_size = max(int(world_size), 1)
    if world_size == 1 or not preserve:
        return batch_size, accumulation_steps, batch_size * accumulation_steps * world_size

    target_global_batch = batch_size * accumulation_steps
    if target_global_batch % world_size == 0:
        per_rank_batch = target_global_batch // world_size
        for candidate in range(min(batch_size, per_rank_batch), 0, -1):
            if per_rank_batch % candidate == 0:
                return candidate, per_rank_batch // candidate, target_global_batch

    per_rank_batch = max(1, min(batch_size, math.ceil(target_global_batch / world_size)))
    per_rank_accumulation = max(1, math.ceil(target_global_batch / (world_size * per_rank_batch)))
    actual_global_batch = per_rank_batch * per_rank_accumulation * world_size
    return per_rank_batch, per_rank_accumulation, actual_global_batch


def _gather_rng_states(data_generator: torch.Generator, world_size: int) -> list[dict] | None:
    local_state = capture_rng_state(data_generator)
    if world_size == 1:
        return None
    states: list[dict | None] = [None] * world_size
    distributed.all_gather_object(states, local_state)
    return states


def _broadcast_object(value, world_size: int):
    if world_size == 1:
        return value
    values = [value]
    distributed.broadcast_object_list(values, src=0)
    return values[0]


def run_training(
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    is_main_process = rank == 0
    set_seed(int(config["seed"]) + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if is_main_process:
            if world_size > 1:
                devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                print(f"Distributed CUDA: world_size={world_size} devices={devices}")
            else:
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
    elif is_main_process:
        print(f"Using device: {device}")

    resume_checkpoint = load_checkpoint(args.resume, device) if args.resume else None
    if resume_checkpoint is not None:
        validate_resume_config(config, resume_checkpoint)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    if is_main_process:
        answer_vocab = load_or_build_answer_vocab(data_cfg, resume_checkpoint)
    if world_size > 1:
        distributed.barrier()
    if not is_main_process:
        answer_vocab = load_or_build_answer_vocab(data_cfg, resume_checkpoint)
    input_pipeline = build_input_pipeline(model_cfg, data_cfg)

    train_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["train_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_train_samples"],
        train=True,
        augmentation=data_cfg.get("augmentation"),
        image_mode=input_pipeline.image_mode,
    )
    val_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["val_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_val_samples"],
        train=False,
        augmentation=data_cfg.get("augmentation"),
        image_mode=input_pipeline.image_mode,
    )
    if is_main_process:
        print(f"answer_vocab={len(answer_vocab)} train_examples={len(train_dataset)} val_examples={len(val_dataset)}")

    configured_batch_size = int(train_cfg["batch_size"])
    configured_accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    preserve_effective_batch_size = bool(train_cfg.get("preserve_effective_batch_size", True))
    batch_size, accumulation_steps, effective_batch_size = effective_batch_plan(
        configured_batch_size,
        configured_accumulation,
        world_size,
        preserve=preserve_effective_batch_size,
    )
    if is_main_process:
        print(
            f"batch_per_gpu={batch_size} accumulation={accumulation_steps} "
            f"global_effective_batch={effective_batch_size}"
        )
        if effective_batch_size != configured_batch_size * configured_accumulation:
            print(
                "note: configured global batch cannot be divided evenly across workers; "
                "using the next valid batch size"
            )

    data_generator = torch.Generator().manual_seed(int(config["seed"]) + rank)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config["seed"]),
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=input_pipeline.collator,
        generator=data_generator,
        sampler=train_sampler,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=input_pipeline.collator,
    )

    model = build_model(model_cfg, answer_vocab_size=len(answer_vocab)).to(device)
    optimizer = build_optimizer(model, train_cfg)
    total_epochs = int(train_cfg["epochs"])
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / accumulation_steps
    )
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs, steps_per_epoch=optimizer_steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("use_amp", False) and device.type == "cuda"))

    runtime_cfg = config.get("runtime", {})
    run_name = runtime_cfg.get("run_name")
    if runtime_cfg.get("use_run_dir", False):
        artifact_dir = create_run_directory(runtime_cfg.get("run_dir", "runs"), run_name) if is_main_process else None
        artifact_dir = Path(_broadcast_object(str(artifact_dir) if artifact_dir else None, world_size))
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
    run_metadata = collect_run_metadata(args.config, device) if is_main_process else {}
    if is_main_process:
        run_metadata["selection_metric"] = train_cfg.get("selection_metric", "vqa_score")
        run_metadata["effective_config"] = config
        run_metadata["run_dir"] = str(artifact_dir)
        run_metadata["distributed"] = {
            "world_size": world_size,
            "rank": rank,
            "batch_per_gpu": batch_size,
            "gradient_accumulation_steps_per_gpu": accumulation_steps,
            "effective_global_batch_size": effective_batch_size,
            "effective_batch_preserved": effective_batch_size == configured_batch_size * configured_accumulation,
        }

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
            rank=rank,
            device=device,
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        history = list(restored.get("history") or [])
        best_metric = float(restored.get("best_metric", best_metric))
        global_step = int(restored.get("global_step", 0))
        epochs_without_improvement = int(restored.get("epochs_without_improvement", 0))
        if is_main_process:
            run_metadata["resumed_from"] = str(Path(args.resume).resolve())
            run_metadata["resumed_epoch"] = int(resume_checkpoint["epoch"])

    if world_size > 1:
        needs_unused_parameter_tracking = (
            train_cfg.get("staged_finetuning", False)
            or train_cfg.get("fixed_finetune_stage") is not None
            or model_cfg.get("freeze_backbones", False)
            or any(not parameter.requires_grad for parameter in model.parameters())
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(needs_unused_parameter_tracking),
        )
    else:
        training_model = model

    if start_epoch > total_epochs:
        raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; configured epochs={total_epochs}.")

    if is_main_process:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_snapshot_path, config)
    resume_wandb_id = None
    if resume_checkpoint is not None:
        checkpoint_metadata = resume_checkpoint.get("metadata") or {}
        if isinstance(checkpoint_metadata, dict):
            resume_wandb_id = (checkpoint_metadata.get("wandb") or {}).get("run_id")
    tracker = (
        init_wandb_tracker(config.get("tracking", {}), config, run_name=run_name, resume_run_id=resume_wandb_id)
        if is_main_process
        else None
    )
    wandb_required = bool(config.get("tracking", {}).get("wandb", {}).get("required", False))
    if is_main_process and wandb_required and not tracker.enabled:
        raise RuntimeError(f"W&B online tracking is required but unavailable: {tracker.reason}")
    if is_main_process and tracker.enabled:
        run_metadata["wandb"] = {"run_id": tracker.run_id, "url": tracker.url}
        print(f"W&B run: {tracker.url}")
    elif is_main_process and tracker.reason:
        run_metadata["wandb"] = {"enabled": False, "reason": tracker.reason}
    if is_main_process:
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
            if is_main_process:
                print(
                    f"epoch={epoch}/{total_epochs} stage={stage} "
                    f"trainable={parameter_counts['trainable']:,}/{parameter_counts['total']:,}"
                )

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                training_model,
                train_loader,
                optimizer,
                device,
                grad_clip_norm=train_cfg["grad_clip_norm"],
                log_every=train_cfg["log_every"],
                use_amp=train_cfg.get("use_amp", False),
                scaler=scaler,
                gradient_accumulation_steps=accumulation_steps,
                label_smoothing=train_cfg.get("label_smoothing", 0.0),
                step_callback=(
                    (lambda payload: tracker.log(payload, step=int(payload["global_step"])))
                    if is_main_process and tracker is not None
                    else None
                ),
                epoch=epoch,
                global_step_start=global_step,
                scheduler=scheduler,
                is_main_process=is_main_process,
            )
            global_step += int(train_metrics["optimizer_steps"])
            val_metrics = (
                evaluate(model, val_loader, device, use_amp=train_cfg.get("use_amp", False))
                if is_main_process
                else None
            )
            val_metrics = _broadcast_object(val_metrics, world_size)
            if selection_metric not in val_metrics:
                available = ", ".join(sorted(val_metrics))
                raise ValueError(f"Unknown selection_metric '{selection_metric}'. Available metrics: {available}")
            selected_value = float(val_metrics[selection_metric])
            step_scheduler(scheduler, selected_value)
            current_lr = _head_learning_rate(optimizer)

            previous_best = best_metric
            is_best = selected_value > previous_best
            if is_best:
                best_metric = selected_value
            meaningful_improvement = selected_value > previous_best + float(
                train_cfg.get("early_stopping_min_delta", 0.0)
            )
            if meaningful_improvement:
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
            distributed_rng_states = _gather_rng_states(data_generator, world_size)
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
                "distributed_rng_states": distributed_rng_states,
            }
            if is_main_process:
                checkpoint_kwargs["model"] = model
                save_checkpoint(latest_path, **checkpoint_kwargs)
                if is_best:
                    save_checkpoint(checkpoint_path, **checkpoint_kwargs)
                    print(f"saved best checkpoint to {checkpoint_path}")

                write_training_history(history_path, history)
                save_training_curves(curves_path, history)
                epoch_log = {
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
                }
                epoch_log.update(_learning_rate_metrics(optimizer))
                tracker.log(epoch_log, step=global_step)
                print(
                    f"train_loss={train_metrics['loss']:.4f} train_vqa={train_metrics['vqa_score']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                    f"val_vqa={val_metrics['vqa_score']:.4f} val_top5={val_metrics['top5_vqa_score']:.4f} "
                    f"lr={current_lr:.2e} latest={latest_path}"
                )

            patience = int(train_cfg.get("early_stopping_patience", 0))
            if patience > 0 and epochs_without_improvement >= patience:
                if is_main_process:
                    print(f"early stopping after {epochs_without_improvement} epochs without improvement")
                break
    finally:
        if is_main_process:
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
            if tracker is not None and tracker.enabled:
                tracker.log_artifact_file(
                    history_path,
                    name=f"{run_name or 'vqa'}-history",
                    artifact_type="training-history",
                )
                tracker.log_artifact_file(
                    curves_path,
                    name=f"{run_name or 'vqa'}-curves",
                    artifact_type="training-curves",
                )
                if config.get("tracking", {}).get("wandb", {}).get("log_checkpoints", False):
                    tracker.log_artifact_file(checkpoint_path, name=f"{run_name or 'vqa'}-best", artifact_type="model")
            if tracker is not None:
                tracker.finish()


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
    device, rank, world_size, local_rank = initialize_distributed(config["device"])
    try:
        run_training(args, config, device, rank, world_size, local_rank)
    finally:
        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
