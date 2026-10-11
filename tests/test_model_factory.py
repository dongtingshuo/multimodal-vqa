from __future__ import annotations

import pytest
import torch

from vqa_project.model import MODEL_REGISTRY, VQAModel, build_model


def base_model_config() -> dict:
    return {
        "answer_vocab_size": 5,
        "text_model_name": "distilbert-base-uncased",
        "hidden_dim": 16,
        "num_attention_heads": 4,
        "dropout": 0.0,
        "freeze_backbones": True,
        "pretrained_cnn": False,
        "mock_backbones": True,
        "mock_hidden_size": 16,
    }


def test_model_factory_defaults_to_cross_attention() -> None:
    model = build_model(base_model_config())
    assert isinstance(model, VQAModel)


def test_model_factory_builds_all_registered_variants() -> None:
    for variant in MODEL_REGISTRY:
        if variant == "beit3_base":
            config = {
                "name": variant,
                "answer_vocab_size": 5,
                "image_size": 16,
                "mock_backbones": True,
                "mock_hidden_size": 8,
            }
        else:
            config = base_model_config()
            config["name"] = variant
        model = build_model(config)
        assert model.__class__ is MODEL_REGISTRY[variant]


def test_beit3_mock_backbone_forward_shape() -> None:
    model = build_model(
        {
            "name": "beit3_base",
            "answer_vocab_size": 5,
            "image_size": 16,
            "mock_backbones": True,
            "mock_hidden_size": 8,
        }
    )
    logits = model(torch.randn(2, 3, 16, 16), torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long))
    assert logits.shape == (2, 5)


def test_model_factory_rejects_unknown_variant() -> None:
    config = base_model_config()
    config["name"] = "unknown"
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model(config)
