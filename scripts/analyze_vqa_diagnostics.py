from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqa_project.analysis.vqa_diagnostics import analyze_vqa_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze answer-vocabulary coverage and full VQA validation errors.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--answer-vocab", required=True)
    parser.add_argument("--official-vqa-score", type=float)
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = analyze_vqa_diagnostics(
        predictions_path=args.predictions,
        questions_path=args.questions,
        annotations_path=args.annotations,
        answer_vocab_path=args.answer_vocab,
        output_dir=args.output_dir,
        official_vqa_score=args.official_vqa_score,
        top_k=args.top_k,
    )
    print(f"wrote json: {paths['json']}")
    print(f"wrote markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
