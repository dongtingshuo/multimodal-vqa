from __future__ import annotations

import torch

from vqa_project.model import MODEL_REGISTRY, build_model


def test_each_model_variant_accepts_mock_batch_and_outputs_logits() -> None:
    batch_size = 2
    answer_vocab_size = 7
    images = torch.rand(batch_size, 3, 32, 32)
    input_ids = torch.ones(batch_size, 6, dtype=torch.long)
    attention_mask = torch.ones(batch_size, 6, dtype=torch.long)

    for variant in MODEL_REGISTRY:
        model = build_model(
            {
                "name": variant,
                "answer_vocab_size": answer_vocab_size,
                "hidden_dim": 16,
                "num_attention_heads": 4,
                "dropout": 0.0,
                "freeze_backbones": True,
                "pretrained_cnn": False,
                "mock_backbones": True,
                "mock_hidden_size": 16,
            }
        )
        logits = model(images, input_ids, attention_mask)
        assert logits.shape == (batch_size, answer_vocab_size)


def test_strong_cross_attention_is_registered_and_has_larger_fusion_head() -> None:
    model = build_model(
        {
            "name": "strong_cross_attention",
            "answer_vocab_size": 7,
            "hidden_dim": 16,
            "num_attention_heads": 4,
            "dropout": 0.0,
            "freeze_backbones": True,
            "pretrained_cnn": False,
            "mock_backbones": True,
            "mock_hidden_size": 16,
        }
    )
    first_linear = next(module for module in model.classifier if isinstance(module, torch.nn.Linear))
    assert first_linear.in_features == 64
    assert "strong_cross_attention" in MODEL_REGISTRY
