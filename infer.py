from __future__ import annotations

import argparse

from vqa_project.answers import AnswerVocab
from vqa_project.config import apply_runtime_overrides, load_config, resolve_checkpoint_config, resolve_device
from vqa_project.engine import load_checkpoint
from vqa_project.inference import predict
from vqa_project.inputs import build_input_pipeline
from vqa_project.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image VQA inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(load_config(args.config), device=args.device)
    device = resolve_device(config["device"], allow_fallback=True)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = resolve_checkpoint_config(config, checkpoint)
    data_cfg = config["data"]
    model_cfg = config["model"]

    answer_vocab = AnswerVocab(
        checkpoint.get("idx_to_answer") or AnswerVocab.load(data_cfg["answer_vocab_path"]).idx_to_answer
    )
    input_pipeline = build_input_pipeline(model_cfg, data_cfg)
    model = build_model(model_cfg, answer_vocab_size=len(answer_vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    results = predict(
        model=model,
        input_pipeline=input_pipeline,
        answer_vocab=answer_vocab,
        image_path=args.image,
        question=args.question,
        device=device,
        image_size=data_cfg["image_size"],
        max_question_length=data_cfg["max_question_length"],
        topk=args.topk or config["infer"]["topk"],
    )
    for answer, probability in results:
        print(f"{answer}\t{probability:.4f}")


if __name__ == "__main__":
    main()
