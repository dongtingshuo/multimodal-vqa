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


def test_vilt_checkpoint_restore_does_not_load_pretrained_weights(monkeypatch) -> None:
    from transformers import ViltConfig

    import vqa_project.model as module

    architecture = ViltConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=32, image_size=32, patch_size=16,
    )
    monkeypatch.setattr(module, "load_vilt_config", lambda _name: architecture)

    def unexpected_download(_name):
        raise AssertionError("Restoring a trained checkpoint must not fetch pretrained weights")

    monkeypatch.setattr(module, "load_vilt_model", unexpected_download)
    restored = build_model({"name": "vilt", "gradient_checkpointing": False}, answer_vocab_size=5, load_pretrained=False)
    assert restored.backbone.config.hidden_size == 16
    assert restored.classifier[-1].out_features == 5
