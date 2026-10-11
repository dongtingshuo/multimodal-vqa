from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa_project.beit3 import BEIT3_VOCAB_SIZE, resolve_beit3_assets
from vqa_project.model import build_model
from vqa_project.training import accumulation_settings_for_global_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure safe BEiT-3 micro-batch sizes on visible GPUs.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--max-question-length", type=int, default=64)
    parser.add_argument("--answer-vocab-size", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default="beit3_batch_benchmark.json")
    return parser.parse_args()


def benchmark_batch(model, batch_size: int, args: argparse.Namespace, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    images = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)
    input_ids = torch.randint(0, BEIT3_VOCAB_SIZE, (batch_size, args.max_question_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    total_steps = args.warmup_steps + args.measure_steps
    elapsed_steps = 0.0
    try:
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images, input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            if step >= args.warmup_steps:
                elapsed_steps += time.perf_counter() - started
        return {
            "batch_size_per_gpu": batch_size,
            "status": "ok",
            "device_name": torch.cuda.get_device_name(device),
            "samples_per_second": batch_size * args.measure_steps / max(elapsed_steps, 1e-9),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory / (1024**3),
        }
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"batch_size_per_gpu": batch_size, "status": "oom"}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The BEiT-3 batch benchmark requires CUDA.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1:
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("Two-GPU benchmark requires an NCCL-enabled PyTorch runtime.")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    model_cfg = {
        "name": "beit3_base",
        "image_size": args.image_size,
        "gradient_checkpointing": True,
        "drop_path_rate": 0.1,
        "cache_dir": args.cache_dir,
    }
    if rank == 0:
        resolve_beit3_assets(args.cache_dir)
    if world_size > 1:
        distributed.barrier()
    model = build_model(model_cfg, answer_vocab_size=args.answer_vocab_size).to(device)
    model.train()
    results = [benchmark_batch(model, size, args, device) for size in args.batch_sizes]
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        distributed.gather_object(results, gathered, dst=0)
        if rank == 0:
            for index, result in enumerate(results):
                result["per_gpu"] = [worker[index] for worker in gathered]
    if rank == 0:
        candidates = [
            result
            for result in results
            if result["status"] == "ok"
            and all(
                max(gpu["peak_allocated_gib"], gpu["peak_reserved_gib"]) < gpu["device_total_gib"] * 0.88
                for gpu in result.get("per_gpu", [result])
            )
        ]
        selected = (
            max(
                candidates,
                key=lambda result: min(
                    gpu["samples_per_second"] for gpu in result.get("per_gpu", [result])
                ),
            )
            if candidates
            else None
        )
        global_batch = 128
        selected_batch = selected["batch_size_per_gpu"] if selected else None
        accumulation = (
            accumulation_settings_for_global_batch(global_batch, selected_batch, world_size)
            if selected_batch
            else None
        )
        report = {
            "model": "BEiT-3 Base",
            "image_size": args.image_size,
            "world_size": world_size,
            "results": results,
            "selected_batch_size_per_gpu": selected_batch,
            "target_global_batch_size": global_batch,
            "accumulation_settings": accumulation,
            "safety_limit": "both peak allocated and reserved memory below 88% of device capacity",
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    if world_size > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
