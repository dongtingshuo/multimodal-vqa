from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed VQA training runs.")
    parser.add_argument("--run", action="append", required=True, help="Run in NAME=ARTIFACT_DIR form.")
    parser.add_argument("--output-dir", default="outputs/experiment_summary")
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--minimum-official-gain", type=float, default=1.0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summarize_run(name: str, artifact_dir: str | Path) -> dict[str, Any]:
    directory = Path(artifact_dir)
    history_path = directory / "training_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing training history: {history_path}")
    with history_path.open(encoding="utf-8", newline="") as file:
        history = list(csv.DictReader(file))
    if not history:
        raise ValueError(f"Empty training history: {history_path}")
    best = max(history, key=lambda row: float(row["val_vqa_score"]))
    metadata = _read_json(directory / "run_metadata.json")
    official = _read_json(directory / "official_vqa_metrics.json")
    return {
        "run": name,
        "best_epoch": int(best["epoch"]),
        "stage": best.get("stage", "unknown"),
        "val_vqa_score": float(best["val_vqa_score"]),
        "val_accuracy": float(best["val_accuracy"]),
        "val_top5_vqa_score": float(best["val_top5_vqa_score"]),
        "official_vqa_score": official.get("overall"),
        "trainable_parameters": int(best.get("trainable_parameters") or 0),
        "total_hours": float(best.get("total_seconds") or 0.0) / 3600.0,
        "device": metadata.get("device"),
        "git_commit": metadata.get("git_commit"),
    }


def write_summary(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "experiment_summary.json"
    csv_path = output / "experiment_summary.csv"
    markdown_path = output / "experiment_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "\n".join(
        f"| {row['run']} | {row['best_epoch']} | {row['stage']} | {row['val_vqa_score']:.4f} | "
        f"{row['official_vqa_score'] if row['official_vqa_score'] is not None else 'n/a'} | "
        f"{row['total_hours']:.2f} |"
        for row in rows
    )
    markdown_path.write_text(
        "# VQA Experiment Summary / VQA \u5b9e\u9a8c\u6c47\u603b\n\n"
        "| Run | Best Epoch | Stage | Internal VQA | Official VQA | Hours |\n"
        "|---|---:|---|---:|---:|---:|\n"
        f"{table}\n",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def write_release_decision(
    rows: list[dict[str, Any]],
    baseline_name: str,
    candidate_name: str,
    minimum_gain: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    by_name = {row["run"]: row for row in rows}
    if baseline_name not in by_name or candidate_name not in by_name:
        raise ValueError("Release gate run names must match supplied --run names.")
    baseline_score = by_name[baseline_name]["official_vqa_score"]
    candidate_score = by_name[candidate_name]["official_vqa_score"]
    if baseline_score is None or candidate_score is None:
        payload = {
            "status": "blocked",
            "reason": "Official VQA scores are required for both runs.",
            "baseline": baseline_name,
            "candidate": candidate_name,
            "baseline_official_vqa": baseline_score,
            "candidate_official_vqa": candidate_score,
            "gain": None,
            "minimum_gain": float(minimum_gain),
            "promote_candidate": False,
        }
    else:
        gain = float(candidate_score) - float(baseline_score)
        promote = gain >= minimum_gain
        payload = {
            "status": "promote" if promote else "reject",
            "reason": None,
            "baseline": baseline_name,
            "candidate": candidate_name,
            "baseline_official_vqa": float(baseline_score),
            "candidate_official_vqa": float(candidate_score),
            "gain": gain,
            "minimum_gain": float(minimum_gain),
            "promote_candidate": promote,
        }
    output = Path(output_dir)
    (output / "release_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    baseline_display = "n/a" if baseline_score is None else f"{float(baseline_score):.2f}"
    candidate_display = "n/a" if candidate_score is None else f"{float(candidate_score):.2f}"
    gain_display = "n/a" if payload["gain"] is None else f"{float(payload['gain']):.2f} points"
    markdown = (
        "# Release Decision / \u53d1\u5e03\u51b3\u7b56\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Baseline: `{baseline_name}` ({baseline_display})\n"
        f"- Candidate: `{candidate_name}` ({candidate_display})\n"
        f"- Gain: `{gain_display}`\n"
        f"- Required gain: `{minimum_gain:.2f} points`\n"
        f"- Promote candidate: `{'yes' if payload['promote_candidate'] else 'no'}`\n"
    )
    if payload["reason"]:
        markdown += f"- Reason: {payload['reason']}\n"
    (output / "release_decision.md").write_text(markdown, encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    runs = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run value: {item}; expected NAME=ARTIFACT_DIR")
        name, directory = item.split("=", maxsplit=1)
        runs.append(summarize_run(name, directory))
    paths = write_summary(runs, args.output_dir)
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")
    if bool(args.baseline) != bool(args.candidate):
        raise ValueError("--baseline and --candidate must be provided together.")
    if args.baseline and args.candidate:
        decision = write_release_decision(
            runs,
            args.baseline,
            args.candidate,
            args.minimum_official_gain,
            args.output_dir,
        )
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
