from __future__ import annotations

import os

from transformers import AutoModel, AutoTokenizer, ViltModel, ViltProcessor


def _local_only() -> bool:
    return os.environ.get("VQA_HF_LOCAL_ONLY") == "1"


def _missing_local_model_message(model_name: str) -> str:
    return (
        f"Model files for '{model_name}' were not found in the local Hugging Face cache. "
        "Run training/inference once in an online environment, or download the model files "
        "before launching the offline Gradio demo."
    )


def load_text_model(model_name: str):
    try:
        return AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        if _local_only():
            raise RuntimeError(_missing_local_model_message(model_name)) from exc
        return AutoModel.from_pretrained(model_name)


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        if _local_only():
            raise RuntimeError(_missing_local_model_message(model_name)) from exc
        return AutoTokenizer.from_pretrained(model_name)


def load_vilt_model(model_name: str):
    try:
        return ViltModel.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        if _local_only():
            raise RuntimeError(_missing_local_model_message(model_name)) from exc
        return ViltModel.from_pretrained(model_name)


def load_vilt_processor(model_name: str):
    try:
        return ViltProcessor.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        if _local_only():
            raise RuntimeError(_missing_local_model_message(model_name)) from exc
        return ViltProcessor.from_pretrained(model_name)
