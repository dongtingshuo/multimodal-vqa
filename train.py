from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vqa_project.answers import AnswerVocab, build_answer_vocab
from vqa_project.config import load_config, resolve_device
from vqa_project.data import VQACollator, VQADataset, default_annotation_path
from vqa_project.engine import evaluate, save_checkpoint, set_seed, train_one_epoch
from vqa_project.hf import load_tokenizer
from vqa_project.model import VQAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet + DistilBERT VQA model.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def load_or_build_answer_vocab(data_cfg: dict) -> AnswerVocab:
    vocab_path = Path(data_cfg["answer_vocab_path"])
    expected_size = int(data_cfg["answer_vocab_size"])
    if vocab_path.exists():
        answer_vocab = AnswerVocab.load(vocab_path)
        if len(answer_vocab) == expected_size:
            return answer_vocab
        print(
            f"answer vocab size changed from {len(answer_vocab)} to {expected_size}; rebuilding {vocab_path}"
        )

    train_annotations = default_annotation_path(data_cfg["root"], data_cfg["train_split"])
    answer_vocab = build_answer_vocab(train_annotations, expected_size)
    answer_vocab.save(vocab_path)
    return answer_vocab


def build_dataloader(dataset, batch_size: int, shuffle: bool, data_cfg: dict, train_cfg: dict, device: torch.device, collator):
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"Using device: {device}")

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    answer_vocab = load_or_build_answer_vocab(data_cfg)
    tokenizer = load_tokenizer(model_cfg["text_model_name"])
    collator = VQACollator(tokenizer, data_cfg["max_question_length"])

    train_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["train_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_train_samples"],
        train=True,
    )
    val_dataset = VQADataset(
        root=data_cfg["root"],
        split=data_cfg["val_split"],
        answer_vocab=answer_vocab,
        image_size=data_cfg["image_size"],
        max_samples=data_cfg["max_val_samples"],
        train=False,
    )
    print(f"answer_vocab={len(answer_vocab)} train_examples={len(train_dataset)} val_examples={len(val_dataset)}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=collator,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        collator=collator,
    )

    model = VQAModel(answer_vocab_size=len(answer_vocab), **model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    checkpoint_path = Path(train_cfg["checkpoint_dir"]) / train_cfg["checkpoint_name"]
    best_accuracy = -1.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=train_cfg["grad_clip_norm"],
            log_every=train_cfg["log_every"],
            use_amp=train_cfg.get("use_amp", False),
        )
        val_metrics = evaluate(model, val_loader, device, use_amp=train_cfg.get("use_amp", False))
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_metrics, config, answer_vocab)
            print(f"saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
