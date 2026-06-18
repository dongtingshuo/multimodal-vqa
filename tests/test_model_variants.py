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
