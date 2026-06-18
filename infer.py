from __future__ import annotations

import argparse

from vqa_project.answers import AnswerVocab
from vqa_project.config import load_config, resolve_device
from vqa_project.engine import load_checkpoint
from vqa_project.hf import load_tokenizer
from vqa_project.inference import predict
from vqa_project.model import VQAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image VQA inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--topk", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"], allow_fallback=True)
    checkpoint = load_checkpoint(args.checkpoint, device)
    data_cfg = config["data"]
    model_cfg = config["model"]

    answer_vocab = AnswerVocab(checkpoint.get("idx_to_answer") or AnswerVocab.load(data_cfg["answer_vocab_path"]).idx_to_answer)
    tokenizer = load_tokenizer(model_cfg["text_model_name"])
    model = VQAModel(answer_vocab_size=len(answer_vocab), **model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])

    results = predict(
        model=model,
        tokenizer=tokenizer,
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
