from __future__ import annotations

import pytest
import torch
from torch import nn

from vqa_project.answers import AnswerVocab
from vqa_project.engine import (
    evaluate,
    load_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
    train_one_epoch,
    vqa_batch_scores,
    vqa_bce_loss,
)


def test_vqa_loss_is_normalized_per_example() -> None:
    logits = torch.zeros(2, 3)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    expected = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / 2
    assert torch.equal(vqa_bce_loss(logits, targets), expected)


def test_vqa_loss_supports_soft_target_smoothing() -> None:
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 0.0]])
    smoothed = torch.tensor([[0.75, 0.25]])
    expected = nn.functional.binary_cross_entropy_with_logits(logits, smoothed, reduction="sum")
    assert torch.equal(vqa_bce_loss(logits, targets, label_smoothing=0.5), expected)


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
    assert checkpoint["format_version"] == 3
    assert checkpoint["config"]["model"]["name"] == "test"
    assert checkpoint["metadata"]["git_commit"] == "abc123"
    assert not path.with_suffix(".pt.tmp").exists()


class TinyVQAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, images, input_ids, attention_mask):
        _ = input_ids, attention_mask
        return self.classifier(images)


def tiny_batches() -> list[dict]:
    batches = []
    for index in range(3):
        batches.append(
            {
                "images": torch.tensor([[float(index), 1.0]]),
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "targets": torch.tensor([[1.0, 0.0]]),
                "labels": torch.tensor([0]),
                "question_ids": [index],
            }
        )
    return batches


def test_gradient_accumulation_steps_on_final_partial_group() -> None:
    model = TinyVQAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = train_one_epoch(
        model,
        tiny_batches(),
        optimizer,
        torch.device("cpu"),
        grad_clip_norm=1.0,
        gradient_accumulation_steps=2,
    )
    assert metrics["optimizer_steps"] == 2


def test_final_partial_accumulation_group_uses_its_actual_size() -> None:
    reference_model = TinyVQAModel()
    accumulated_model = TinyVQAModel()
    accumulated_model.load_state_dict(reference_model.state_dict())
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    final_batch = tiny_batches()[:1]

    train_one_epoch(
        reference_model,
        final_batch,
        reference_optimizer,
        torch.device("cpu"),
        grad_clip_norm=0.0,
        gradient_accumulation_steps=1,
    )
    train_one_epoch(
        accumulated_model,
        final_batch,
        accumulated_optimizer,
        torch.device("cpu"),
        grad_clip_norm=0.0,
        gradient_accumulation_steps=2,
    )

    for reference, accumulated in zip(reference_model.parameters(), accumulated_model.parameters()):
        assert torch.allclose(reference, accumulated)


def test_evaluate_can_collect_official_submission_records() -> None:
    model = TinyVQAModel()
    records: list[dict] = []
    batches = tiny_batches()[:1]
    batches.append(
        {
            **tiny_batches()[1],
            "targets": torch.zeros(1, 2),
            "labels": torch.tensor([-1]),
        }
    )
    metrics = evaluate(
        model,
        batches,
        torch.device("cpu"),
        answer_vocab=AnswerVocab(["yes", "no"]),
        prediction_records=records,
    )
    assert [record["question_id"] for record in records] == [0, 1]
    assert metrics["evaluated_examples"] == 1


def test_restore_training_checkpoint_restores_state(tmp_path) -> None:
    model = TinyVQAModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(123)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=2,
        metrics={"vqa_score": 0.4},
        config={"model": {"name": "test"}},
        answer_vocab=AnswerVocab(["yes", "no"]),
        training_state={"best_metric": 0.4, "global_step": 7},
        data_generator=generator,
    )
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    restored = restore_training_checkpoint(checkpoint, model, optimizer, None, None, generator)
    assert restored["best_metric"] == 0.4
    assert restored["global_step"] == 7


def test_restore_training_checkpoint_moves_rng_states_to_cpu(tmp_path, monkeypatch) -> None:
    class DeviceMappedState:
        def __init__(self, state: torch.Tensor) -> None:
            self.state = state
            self.cpu_called = False

        def cpu(self) -> torch.Tensor:
            self.cpu_called = True
            return self.state

    model = TinyVQAModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(123)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=2,
        metrics={"vqa_score": 0.4},
        config={"model": {"name": "test"}},
        answer_vocab=AnswerVocab(["yes", "no"]),
        data_generator=generator,
    )
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    torch_state = DeviceMappedState(checkpoint["rng_state"]["torch"])
    cuda_state = DeviceMappedState(checkpoint["rng_state"]["torch"])
    generator_state = DeviceMappedState(checkpoint["rng_state"]["data_generator"])
    checkpoint["rng_state"]["torch"] = torch_state
    checkpoint["rng_state"]["cuda"] = [cuda_state]
    checkpoint["rng_state"]["data_generator"] = generator_state
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)

    restore_training_checkpoint(checkpoint, model, optimizer, None, None, generator)

    assert torch_state.cpu_called
    assert cuda_state.cpu_called
    assert generator_state.cpu_called
