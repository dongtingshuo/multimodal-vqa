from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa_project.data import default_annotation_path, default_question_path


def split_question_count(root: Path, split: str) -> int:
    with default_question_path(root, split).open("r", encoding="utf-8") as source:
        question_ids = {int(item["question_id"]) for item in json.load(source)["questions"]}
    with default_annotation_path(root, split).open("r", encoding="utf-8") as source:
        annotations = json.load(source)["annotations"]
    return sum(int(item["question_id"]) in question_ids for item in annotations)


def estimate_runtime(
    config_path: str | Path,
    benchmark_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    kaggle_session_hours: float = 10.5,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    batch_size = benchmark.get("selected_batch_size_per_gpu")
    if batch_size is None:
        raise RuntimeError("Cannot estimate training time because no micro-batch passed the GPU benchmark.")
    selected = next(
        (
            row
            for row in benchmark.get("results", [])
            if int(row.get("batch_size_per_gpu", -1)) == int(batch_size) and row.get("status") == "ok"
        ),
        None,
    )
    if selected is None:
        raise ValueError("Selected micro-batch is missing from the successful benchmark results.")
    workers = selected.get("per_gpu") or [selected]
    rates = [float(worker["samples_per_second"]) for worker in workers]
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("Benchmark throughput must be positive on every participating GPU.")

    root = Path(data_root)
    train_examples = split_question_count(root, str(config["data"].get("train_split", "train")))
    val_examples = split_question_count(root, str(config["data"].get("val_split", "val")))
    if config["data"].get("max_train_samples") is not None:
        train_examples = min(train_examples, int(config["data"]["max_train_samples"]))
    if config["data"].get("max_val_samples") is not None:
        val_examples = min(val_examples, int(config["data"]["max_val_samples"]))

    visual_genome_candidates = 0
    visual_genome_sampled = 0
    visual_genome_path = config["data"].get("visual_genome_qa_path")
    if visual_genome_path:
        vg_payload = json.loads(Path(visual_genome_path).read_text(encoding="utf-8"))
        visual_genome_candidates = len(vg_payload.get("qas", []))
        ratio = float(config["data"].get("visual_genome_sample_ratio", 0.25))
        visual_genome_sampled = min(visual_genome_candidates, math.ceil(train_examples * ratio))
    training_examples_per_epoch = train_examples + visual_genome_sampled

    train_seconds = training_examples_per_epoch / sum(rates)
    validation_seconds = val_examples / min(rates)
    model_only_epoch_seconds = train_seconds + validation_seconds
    planning_epoch_seconds = model_only_epoch_seconds * 1.5
    epochs = int(config["train"]["epochs"])
    total_hours = planning_epoch_seconds * epochs / 3600
    payload = {
        "world_size": int(benchmark["world_size"]),
        "selected_batch_size_per_gpu": int(batch_size),
        "target_global_batch_size": int(benchmark["target_global_batch_size"]),
        "per_gpu_samples_per_second": rates,
        "train_examples_per_epoch": train_examples,
        "visual_genome_candidate_examples": visual_genome_candidates,
        "visual_genome_sampled_per_epoch": visual_genome_sampled,
        "total_training_examples_per_epoch": training_examples_per_epoch,
        "validation_examples_per_epoch": val_examples,
        "epochs_configured": epochs,
        "model_only_epoch_hours": model_only_epoch_seconds / 3600,
        "planning_epoch_hours_with_1_5x_overhead": planning_epoch_seconds / 3600,
        "planning_total_hours": total_hours,
        "estimated_sessions_at_configured_limit": math.ceil(total_hours / kaggle_session_hours),
        "session_hours_assumption": kaggle_session_hours,
        "estimate_note": (
            "Approximation from short GPU-only forward/backward throughput; the 1.5x planning factor is a buffer, "
            "not a guarantee. Data loading, checkpoint I/O, startup, and early stopping may change wall time."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate BEiT-3 epoch and session count from measured GPU throughput.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--session-hours", type=float, default=10.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimate_runtime(args.config, args.benchmark, args.data_root, args.output, args.session_hours)


if __name__ == "__main__":
    main()
