from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def create_replication_config(source_path: str | Path, output_path: str | Path) -> dict:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("The replication config must not overwrite its source experiment config.")
    with source_path.open("r", encoding="utf-8") as source_file:
        config = yaml.safe_load(source_file)
    if config.get("model", {}).get("name") != "beit3_base" or int(config.get("seed", -1)) != 42:
        raise ValueError("The source must be a seed-42 BEiT-3 A or B experiment config.")

    config["seed"] = 123
    config["train"]["checkpoint_dir"] = "/kaggle/working/multimodal-vqa/beit3-c-seed123-t4x2"
    config.setdefault("runtime", {})["run_name"] = "beit3-vqa-c-seed123-t4x2"
    wandb_config = config.setdefault("tracking", {}).setdefault("wandb", {})
    tags = list(wandb_config.get("tags", []))
    tags = [tag for tag in tags if tag not in {"seed42", "experiment-a", "experiment-b"}]
    for tag in ("seed123", "experiment-c"):
        if tag not in tags:
            tags.append(tag)
    wandb_config["tags"] = tags

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the seed-123 BEiT-3 replication config from A or B.")
    parser.add_argument("--source-config", required=True, help="Winning seed-42 A or B YAML config.")
    parser.add_argument(
        "--output-config",
        default="configs/kaggle_beit3_c_seed123_t4x2.yaml",
        help="Output path for the independent replication config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = create_replication_config(args.source_config, args.output_config)
    data_source = "visual_genome" if config["data"].get("visual_genome_qa_path") else "vqa_v2_only"
    print(f"created {args.output_config}: seed={config['seed']} data_source={data_source}")


if __name__ == "__main__":
    main()
