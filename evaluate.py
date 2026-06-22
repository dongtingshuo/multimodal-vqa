from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from vqa_project.answers import AnswerVocab
from vqa_project.config import load_config, resolve_checkpoint_config, resolve_device
from vqa_project.data import VQACollator, VQADataset
from vqa_project.engine import evaluate, load_checkpoint
from vqa_project.hf import load_tokenizer
from vqa_project.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained VQA model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"], allow_fallback=True)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = resolve_checkpoint_config(config, checkpoint)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    answer_vocab = AnswerVocab(
        checkpoint.get("idx_to_answer") or AnswerVocab.load(data_cfg["answer_vocab_path"]).idx_to_answer
    )
    tokenizer = load_tokenizer(model_cfg["text_model_name"])
    collator = VQACollator(tokenizer, data_cfg["max_question_length"])

    val_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["val_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_val_samples"],
        train=False,
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = build_model(model_cfg, answer_vocab_size=len(answer_vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, val_loader, device, use_amp=train_cfg.get("use_amp", False))
    print(
        f"val_loss={metrics['loss']:.4f} "
        f"val_acc={metrics['accuracy']:.4f} "
        f"val_vqa={metrics['vqa_score']:.4f} "
        f"val_top5_vqa={metrics['top5_vqa_score']:.4f}"
    )


if __name__ == "__main__":
    main()
