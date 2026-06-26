from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training runs stored under a runs directory.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="runs/summary.csv")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    history_path = run_dir / "training_history.csv"
    if not history_path.is_file():
        return None
    with history_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("val_vqa_score") or 0.0))
    metadata = _read_json(run_dir / "run_metadata.json")
    summary = _read_json(run_dir / "run_summary.json")
    return {
        "run_dir": str(run_dir),
        "best_epoch": int(best["epoch"]),
        "stage": best.get("stage", ""),
        "val_loss": float(best.get("val_loss") or 0.0),
        "val_accuracy": float(best.get("val_accuracy") or 0.0),
        "val_vqa_score": float(best.get("val_vqa_score") or 0.0),
        "val_top5_vqa_score": float(best.get("val_top5_vqa_score") or 0.0),
        "learning_rate": float(best.get("learning_rate") or 0.0),
        "total_hours": float(best.get("total_seconds") or 0.0) / 3600.0,
        "checkpoint": (metadata.get("artifacts") or summary.get("artifacts") or {}).get("checkpoint", ""),
        "git_commit": metadata.get("git_commit", ""),
        "wandb_url": (metadata.get("wandb") or {}).get("url", ""),
    }


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = [
        row
        for row in (summarize_run(path) for path in sorted(runs_dir.iterdir()) if path.is_dir())
        if row is not None
    ]
    rows.sort(key=lambda row: row["val_vqa_score"], reverse=True)
    write_rows(rows, Path(args.output))
    print(f"summarized {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
