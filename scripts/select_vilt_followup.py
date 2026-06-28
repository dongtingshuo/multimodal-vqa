from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the single-variable second ViLT experiment.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--base-config", default="configs/kaggle_vilt.yaml")
    parser.add_argument("--output-config", default="outputs/kaggle_vilt_followup.yaml")
    parser.add_argument("--decision-output", default="outputs/kaggle_vilt_followup_decision.json")
    return parser.parse_args()


def read_history(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open(encoding="utf-8") as file:
        return [
            {
                key: float(value)
                for key, value in row.items()
                if key in {"train_vqa_score", "val_vqa_score", "val_accuracy"}
            }
            for row in csv.DictReader(file)
        ]


def select_followup(rows: list[dict[str, float]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Training history is empty.")
    best = max(rows, key=lambda row: row["val_vqa_score"])
    values = [row["val_vqa_score"] for row in rows]
    unstable = any(not math.isfinite(value) for value in values) or sum(
        previous - current > 0.03 for previous, current in zip(values, values[1:])
    ) >= 2
    max_gap = max(row["train_vqa_score"] - row["val_vqa_score"] for row in rows)

    if best["val_accuracy"] >= 0.55 and best["val_vqa_score"] >= 0.65:
        return {"branch": "replicate", "reason": "Run 1 passed both quality gates.", "seed": 1337}
    if unstable:
        return {
            "branch": "lower_backbone_lr",
            "reason": "Validation VQA score showed repeated drops greater than 0.03.",
            "seed": 42,
            "backbone_lr": 1e-5,
        }
    if max_gap > 0.10:
        return {
            "branch": "last_six_layers",
            "reason": f"Maximum train-validation VQA gap was {max_gap:.4f}, above 0.10.",
            "seed": 42,
            "fixed_finetune_stage": "partial",
            "trainable_vilt_layers": 6,
        }
    return {
        "branch": "higher_backbone_lr",
        "reason": "Run 1 was stable without a large generalization gap but remained under the quality gates.",
        "seed": 42,
        "backbone_lr": 3e-5,
    }


def build_followup_config(base_config: dict, decision: dict[str, object]) -> dict:
    config = base_config
    config["seed"] = int(decision["seed"])
    if "backbone_lr" in decision:
        config["train"]["backbone_lr"] = float(decision["backbone_lr"])
    if "fixed_finetune_stage" in decision:
        config["train"]["fixed_finetune_stage"] = decision["fixed_finetune_stage"]
        config["train"]["trainable_vilt_layers"] = int(decision["trainable_vilt_layers"])
    branch = str(decision["branch"])
    config["runtime"]["run_name"] = f"kaggle-vilt-{branch}"
    checkpoint_parent = Path(config["train"]["checkpoint_dir"]).parent
    config["train"]["checkpoint_dir"] = str(checkpoint_parent / f"vilt-{branch}")
    config["tracking"]["wandb"]["tags"] = ["kaggle", "vilt", "coco2014-vqa", "run2", branch]
    return config


def main() -> None:
    args = parse_args()
    decision = select_followup(read_history(args.history))
    config = build_followup_config(load_config(args.base_config), decision)
    config_path = Path(args.output_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    decision_path = Path(args.decision_output)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"followup_config={config_path}")


if __name__ == "__main__":
    main()
