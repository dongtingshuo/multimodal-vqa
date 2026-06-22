from __future__ import annotations

import pytest
import torch
from torch import nn

from vqa_project.answers import AnswerVocab
from vqa_project.engine import load_checkpoint, save_checkpoint, vqa_batch_scores, vqa_bce_loss


def test_vqa_loss_is_normalized_per_example() -> None:
    logits = torch.zeros(2, 3)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    expected = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / 2
    assert torch.equal(vqa_bce_loss(logits, targets), expected)


def test_vqa_scores_use_soft_target_credit() -> None:
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.0, 1.0, 5.0]])
    targets = torch.tensor([[0.3, 1.0, 0.0], [0.0, 0.6, 0.9]])
    scores = vqa_batch_scores(logits, targets)
    assert scores["vqa_score"] == pytest.approx(1.2)
    assert scores["top5_vqa_score"] == pytest.approx(1.9)


def test_checkpoint_format_records_config_and_metadata(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=3,
        metrics={"vqa_score": 0.5},
        config={"model": {"name": "test"}},
        answer_vocab=AnswerVocab(["yes", "no"]),
        metadata={"git_commit": "abc123"},
    )
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["format_version"] == 2
    assert checkpoint["config"]["model"]["name"] == "test"
    assert checkpoint["metadata"]["git_commit"] == "abc123"
