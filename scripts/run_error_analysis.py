from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.analysis import analyze_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight VQA error-analysis report.")
    parser.add_argument("--predictions", default="examples/toy_vqa_demo/toy_predictions.jsonl")
    parser.add_argument("--output-dir", default="outputs/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = analyze_errors(args.predictions, args.output_dir)
    print(f"wrote json: {paths['json']}")
    print(f"wrote markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
