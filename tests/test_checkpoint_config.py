from __future__ import annotations

from vqa_project.config import load_config, resolve_checkpoint_config


def test_checkpoint_owns_model_and_preprocessing_configuration() -> None:
    runtime = load_config("configs/default.yaml")
    runtime["data"]["root"] = "/local/data"
    runtime["model"]["name"] = "image_only"
    checkpoint = {
        "config": {
            "model": {"hidden_dim": 256, "text_model_name": "stored-text-model"},
            "data": {"image_size": 384, "max_question_length": 48, "answer_vocab_size": 42},
        }
    }

    resolved = resolve_checkpoint_config(runtime, checkpoint)

    assert resolved["model"]["name"] == "cross_attention"
    assert resolved["model"]["hidden_dim"] == 256
    assert resolved["model"]["text_model_name"] == "stored-text-model"
    assert resolved["data"]["image_size"] == 384
    assert resolved["data"]["max_question_length"] == 48
    assert resolved["data"]["answer_vocab_size"] == 42
    assert resolved["data"]["root"] == "/local/data"


def test_runtime_config_is_used_for_legacy_checkpoint_without_config() -> None:
    runtime = load_config("configs/default.yaml")
    resolved = resolve_checkpoint_config(runtime, {})
    assert resolved == runtime
    assert resolved is not runtime
