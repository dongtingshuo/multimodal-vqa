from __future__ import annotations

from transformers import AutoModel, AutoTokenizer


def load_text_model(model_name: str):
    try:
        return AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoModel.from_pretrained(model_name)


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(model_name)
