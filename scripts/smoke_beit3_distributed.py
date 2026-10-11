from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa_project.answers import AnswerVocab
from vqa_project.beit3 import BEIT3_VOCAB_SIZE, resolve_beit3_assets
from vqa_project.engine import (
    capture_rng_state,
    load_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
    set_seed,
    vqa_bce_loss,
)
from vqa_project.model import build_model
from vqa_project.training import build_optimizer


def _nested_equal(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _nested_equal(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return left == right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-GPU BEiT-3 forward/backward/checkpoint smoke test.")
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--question-length", type=int, default=32)
    parser.add_argument("--answer-vocab-size", type=int, default=32)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--expected-gpus", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The BEiT-3 distributed smoke test requires CUDA.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} workers, got WORLD_SIZE={world_size}.")
    if torch.cuda.device_count() < args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs, got {torch.cuda.device_count()}.")

    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    set_seed(42 + rank)

    model_cfg = {
        "name": "beit3_base",
        "image_size": args.image_size,
        "drop_path_rate": 0.1,
        "gradient_checkpointing": True,
        "cache_dir": args.cache_dir,
    }
    if rank == 0:
        resolve_beit3_assets(args.cache_dir)
    distributed.barrier()

    model = build_model(model_cfg, answer_vocab_size=args.answer_vocab_size).to(device)
    model.set_finetune_stage("full")
    permanently_frozen = set(model.permanently_frozen_parameter_names)
    for name, parameter in model.named_parameters():
        if name in permanently_frozen:
            parameter.requires_grad_(False)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = build_optimizer(
        model,
        {
            "lr": 3e-5,
            "backbone_lr": 3e-5,
            "head_lr": 6e-4,
            "betas": (0.9, 0.98),
            "weight_decay": 0.01,
        },
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    data_generator = torch.Generator().manual_seed(1000 + rank)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    for step in range(2):
        images = torch.rand(args.batch_size_per_gpu, 3, args.image_size, args.image_size, device=device)
        input_ids = torch.randint(
            3,
            BEIT3_VOCAB_SIZE,
            (args.batch_size_per_gpu, args.question_length),
            device=device,
        )
        input_ids[:, 0] = 0
        input_ids[:, -1] = 2
        attention_mask = torch.ones_like(input_ids)
        targets = torch.rand(args.batch_size_per_gpu, args.answer_vocab_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = ddp_model(images, input_ids, attention_mask)
            loss = vqa_bce_loss(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Rank {rank} produced a non-finite loss at update {step + 1}.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        print(
            f"rank={rank} gpu={torch.cuda.get_device_name(device)} update={step + 1}/2 "
            f"loss={loss.item():.5f} parameters={parameter_count:,}",
            flush=True,
        )

    probe = model.head[-1].weight[0, 0].detach().clone()
    minimum, maximum = probe.clone(), probe.clone()
    distributed.all_reduce(minimum, op=distributed.ReduceOp.MIN)
    distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX)
    if not torch.allclose(minimum, maximum, atol=1e-6, rtol=0):
        raise RuntimeError("DDP workers diverged after synchronized optimizer updates.")

    rng_states: list[dict | None] = [None] * world_size
    distributed.all_gather_object(rng_states, capture_rng_state(data_generator))
    expected_optimizer_state = copy.deepcopy(optimizer.state_dict())
    expected_scaler_state = copy.deepcopy(scaler.state_dict())
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = tempfile.TemporaryDirectory(prefix="beit3-smoke-", dir=output_root) if rank == 0 else None
    checkpoint_path_value = [str(Path(temporary_dir.name) / "roundtrip.pt") if temporary_dir else None]
    distributed.broadcast_object_list(checkpoint_path_value, src=0)
    checkpoint_path = Path(checkpoint_path_value[0])
    if rank == 0:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={"loss": float(loss.item())},
            config={"model": model_cfg},
            answer_vocab=AnswerVocab([str(index) for index in range(args.answer_vocab_size)], normalization="vqa_v2"),
            training_state={"global_step": 2, "epoch_complete": False, "batch_in_epoch": 2},
            data_generator=data_generator,
            distributed_rng_states=rng_states,
        )
        print(f"checkpoint saved: {checkpoint_path.stat().st_size:,} bytes", flush=True)
    distributed.barrier()

    checkpoint = load_checkpoint(checkpoint_path, device)
    expected_bias = model.head[-1].bias.detach().clone()
    model.head[-1].bias.data.add_(1.0)
    next(iter(optimizer.state.values()))["exp_avg"].add_(1.0)
    scaler.update(new_scale=scaler.get_scale() * 2)
    random.random()
    torch.rand(1, device=device)
    torch.rand(1, generator=data_generator)
    restored = restore_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        data_generator=data_generator,
        rank=rank,
        device=device,
    )
    if not torch.equal(model.head[-1].bias, expected_bias):
        raise RuntimeError("Checkpoint restore did not recover the trained model parameters.")
    if int(restored.get("global_step", -1)) != 2:
        raise RuntimeError("Checkpoint restore did not recover the optimizer-step cursor.")
    if not _nested_equal(optimizer.state_dict(), expected_optimizer_state):
        raise RuntimeError("Checkpoint restore did not recover optimizer state.")
    if not _nested_equal(scaler.state_dict(), expected_scaler_state):
        raise RuntimeError("Checkpoint restore did not recover AMP scaler state.")
    restored_rng = capture_rng_state(data_generator)
    expected_rng = rng_states[rank]
    if not _nested_equal(restored_rng, expected_rng):
        raise RuntimeError("Checkpoint restore did not recover this rank's RNG and data-generator state.")
    print(f"rank={rank} checkpoint restore passed at global_step={restored['global_step']}", flush=True)
    distributed.barrier()
    if temporary_dir is not None:
        temporary_dir.cleanup()
    distributed.destroy_process_group()
    if rank == 0:
        print("BEiT-3 two-GPU forward, backward, synchronization, save, and restore smoke passed.", flush=True)


if __name__ == "__main__":
    main()
