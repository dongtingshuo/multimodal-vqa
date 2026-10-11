from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.estimate_beit3_runtime import estimate_runtime
from scripts.prepare_beit3_runtime_config import prepare_runtime_config


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": 480,
                    "max_train_samples": None,
                    "max_val_samples": None,
                    "visual_genome_qa_path": None,
                },
                "model": {"name": "beit3_base"},
                "train": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 128,
                    "preserve_effective_batch_size": True,
                    "epochs": 10,
                },
                "runtime": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_benchmark(path: Path, world_size: int = 2, selected_batch: int = 2) -> None:
    workers = [
        {
            "status": "ok",
            "samples_per_second": 4.0,
            "peak_allocated_gib": 10.0,
            "peak_reserved_gib": 11.0,
            "device_total_gib": 16.0,
        }
        for _ in range(world_size)
    ]
    path.write_text(
        json.dumps(
            {
                "world_size": world_size,
                "image_size": 480,
                "target_global_batch_size": 128,
                "selected_batch_size_per_gpu": selected_batch,
                "results": [
                    {
                        "batch_size_per_gpu": selected_batch,
                        "status": "ok",
                        "per_gpu": workers,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_runtime_config_applies_benchmarked_batch_and_preserves_global_batch(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "runtime.yaml"
    _write_config(source)
    _write_benchmark(benchmark)

    config = prepare_runtime_config(source, benchmark, output, expected_world_size=2)

    assert config["train"]["batch_size"] == 2
    assert config["train"]["gradient_accumulation_steps"] == 64
    assert config["train"]["batch_size"] * config["train"]["gradient_accumulation_steps"] == 128
    assert config["runtime"]["batch_benchmark"]["runtime_gradient_accumulation_steps_per_rank"] == 32
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == config


def test_runtime_config_rejects_unsafe_or_mismatched_benchmark(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "runtime.yaml"
    _write_config(source)
    _write_benchmark(benchmark, world_size=1)
    with pytest.raises(ValueError, match="world_size"):
        prepare_runtime_config(source, benchmark, output, expected_world_size=2)

    _write_benchmark(benchmark)
    report = json.loads(benchmark.read_text(encoding="utf-8"))
    report["results"][0]["per_gpu"][0]["peak_reserved_gib"] = 15.0
    benchmark.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="memory safety"):
        prepare_runtime_config(source, benchmark, output, expected_world_size=2)


def _write_vqa_split(root: Path, split: str, question_ids: list[int]) -> None:
    coco_split = "train2014" if split == "train" else "val2014"
    questions_path = root / f"v2_OpenEnded_mscoco_{coco_split}_questions.json"
    annotations_path = root / f"v2_mscoco_{coco_split}_annotations.json"
    questions_path.write_text(
        json.dumps({"questions": [{"question_id": question_id} for question_id in question_ids]}),
        encoding="utf-8",
    )
    annotations_path.write_text(
        json.dumps({"annotations": [{"question_id": question_id} for question_id in question_ids]}),
        encoding="utf-8",
    )


def test_runtime_estimate_counts_vqa_and_screened_vg_examples(tmp_path: Path) -> None:
    data_root = tmp_path / "vqa"
    data_root.mkdir()
    _write_vqa_split(data_root, "train", [1, 2, 3, 4])
    _write_vqa_split(data_root, "val", [10, 11])
    vg_manifest = tmp_path / "vg.json"
    vg_manifest.write_text(json.dumps({"qas": [{"qa_id": index} for index in range(10)]}), encoding="utf-8")
    config = tmp_path / "runtime.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_split": "train",
                    "val_split": "val",
                    "max_train_samples": None,
                    "max_val_samples": None,
                    "visual_genome_qa_path": str(vg_manifest),
                    "visual_genome_sample_ratio": 0.25,
                },
                "train": {"epochs": 2},
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.json"
    _write_benchmark(benchmark, selected_batch=2)
    report = json.loads(benchmark.read_text(encoding="utf-8"))
    report["target_global_batch_size"] = 128
    benchmark.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "estimate.json"

    estimate = estimate_runtime(config, benchmark, data_root, output, kaggle_session_hours=0.001)

    assert estimate["train_examples_per_epoch"] == 4
    assert estimate["visual_genome_candidate_examples"] == 10
    assert estimate["visual_genome_sampled_per_epoch"] == 1
    assert estimate["total_training_examples_per_epoch"] == 5
    assert estimate["validation_examples_per_epoch"] == 2
    assert estimate["estimated_sessions_at_configured_limit"] >= 1
    assert json.loads(output.read_text(encoding="utf-8")) == estimate
