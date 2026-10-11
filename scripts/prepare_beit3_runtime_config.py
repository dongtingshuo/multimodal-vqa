from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa_project.training import accumulation_settings_for_global_batch


def prepare_runtime_config(
    config_path: str | Path,
    benchmark_path: str | Path,
    output_path: str | Path,
    expected_world_size: int | None = None,
) -> dict:
    config_path = Path(config_path)
    benchmark_path = Path(benchmark_path)
    output_path = Path(output_path)
    if output_path.resolve() in {config_path.resolve(), benchmark_path.resolve()}:
        raise ValueError("Runtime config output must not overwrite the source config or benchmark report.")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if config.get("model", {}).get("name") != "beit3_base":
        raise ValueError("Runtime batch selection is only supported for BEiT-3 configurations.")
    if not config.get("train", {}).get("preserve_effective_batch_size", False):
        raise ValueError("BEiT-3 runtime batch selection requires preserve_effective_batch_size: true.")

    world_size = int(report.get("world_size", 0))
    if world_size < 1 or (expected_world_size is not None and world_size != expected_world_size):
        raise ValueError(
            f"Batch benchmark world_size={world_size} does not match expected world_size={expected_world_size}."
        )
    if int(report.get("image_size", -1)) != int(config.get("data", {}).get("image_size", -2)):
        raise ValueError("Batch benchmark image size does not match the training config.")

    selected_batch = report.get("selected_batch_size_per_gpu")
    if selected_batch is None:
        raise RuntimeError("No BEiT-3 micro-batch passed the benchmark memory safety threshold.")
    selected_batch = int(selected_batch)
    global_batch = int(config["train"]["batch_size"]) * int(config["train"]["gradient_accumulation_steps"])
    if int(report.get("target_global_batch_size", -1)) != global_batch:
        raise ValueError("Batch benchmark target does not match the experiment's configured global batch size.")
    selected_result = next(
        (
            result
            for result in report.get("results", [])
            if int(result.get("batch_size_per_gpu", -1)) == selected_batch and result.get("status") == "ok"
        ),
        None,
    )
    if selected_result is None:
        raise ValueError("The selected micro-batch does not have a successful benchmark result.")
    workers = selected_result.get("per_gpu") or [selected_result]
    if len(workers) != world_size:
        raise ValueError("The selected micro-batch does not contain a result for every participating GPU.")
    for worker in workers:
        total_memory = float(worker.get("device_total_gib", 0))
        peak_memory = max(
            float(worker.get("peak_allocated_gib", float("inf"))),
            float(worker.get("peak_reserved_gib", float("inf"))),
        )
        if worker.get("status") != "ok" or total_memory <= 0 or peak_memory >= total_memory * 0.88:
            raise ValueError("The selected micro-batch did not meet the per-GPU memory safety threshold.")
    settings = accumulation_settings_for_global_batch(global_batch, selected_batch, world_size)

    config["train"]["batch_size"] = selected_batch
    config["train"]["gradient_accumulation_steps"] = settings["config_gradient_accumulation_steps"]
    config.setdefault("runtime", {})["batch_benchmark"] = {
        "report": str(benchmark_path),
        "world_size": world_size,
        "selected_batch_size_per_gpu": selected_batch,
        "target_global_batch_size": global_batch,
        "runtime_gradient_accumulation_steps_per_rank": settings[
            "runtime_gradient_accumulation_steps_per_rank"
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the measured safe BEiT-3 batch size to a run config.")
    parser.add_argument("--config", required=True, help="Source BEiT-3 experiment YAML.")
    parser.add_argument("--benchmark", required=True, help="JSON output from benchmark_beit3_batch.py.")
    parser.add_argument("--output", required=True, help="Path for the generated runtime YAML.")
    parser.add_argument("--world-size", type=int, required=True, help="Expected number of CUDA workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = prepare_runtime_config(args.config, args.benchmark, args.output, args.world_size)
    train = config["train"]
    runtime = config["runtime"]["batch_benchmark"]
    print(
        f"runtime config: batch_per_gpu={train['batch_size']} "
        f"accumulation_per_gpu={runtime['runtime_gradient_accumulation_steps_per_rank']} "
        f"global_batch={runtime['target_global_batch_size']}"
    )


if __name__ == "__main__":
    main()
