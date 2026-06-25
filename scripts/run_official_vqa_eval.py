from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official VQA evaluation toolkit.")
    parser.add_argument("--toolkit-root", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="outputs/official_vqa_metrics.json")
    return parser.parse_args()


def load_toolkit(toolkit_root: str | Path):
    root = Path(toolkit_root)
    helper_path = root / "PythonHelperTools"
    evaluation_path = root / "PythonEvaluationTools"
    if not helper_path.is_dir() or not evaluation_path.is_dir():
        raise FileNotFoundError(
            "Expected PythonHelperTools and PythonEvaluationTools under the official VQA toolkit root."
        )
    sys.path[:0] = [str(helper_path), str(evaluation_path)]
    from vqaEvaluation.vqaEval import VQAEval
    from vqaTools.vqa import VQA

    return VQA, VQAEval


def main() -> None:
    args = parse_args()
    VQA, VQAEval = load_toolkit(args.toolkit_root)
    vqa = VQA(args.annotations, args.questions)
    results = vqa.loadRes(args.predictions, args.questions)
    evaluator = VQAEval(vqa, results, n=2)
    evaluator.evaluate()
    payload = {
        "overall": evaluator.accuracy["overall"],
        "per_question_type": evaluator.accuracy["perQuestionType"],
        "per_answer_type": evaluator.accuracy["perAnswerType"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
