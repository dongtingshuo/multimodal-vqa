from __future__ import annotations

import sys
import types

import pytest
import torch
import yaml
from torch import nn
from torchvision import transforms

from scripts.create_beit3_replication_config import create_replication_config
from vqa_project import beit3
from vqa_project.data import build_image_transform
from vqa_project.model import build_model


@pytest.mark.parametrize(
    ("source_config", "uses_visual_genome"),
    [
        ("configs/kaggle_beit3_a_t4x2.yaml", False),
        ("configs/kaggle_beit3_b_vg_t4x2.yaml", True),
    ],
)
def test_replication_config_copies_winning_recipe_and_changes_only_seed_paths_tags(
    tmp_path,
    source_config: str,
    uses_visual_genome: bool,
) -> None:
    output_path = tmp_path / "replication.yaml"
    generated = create_replication_config(source_config, output_path)
    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert generated == written
    assert generated["seed"] == 123
    assert ("visual_genome_qa_path" in generated["data"]) is uses_visual_genome
    assert generated["train"]["checkpoint_dir"].endswith("beit3-c-seed123-t4x2")
    assert generated["runtime"]["run_name"] == "beit3-vqa-c-seed123-t4x2"
    assert "seed123" in generated["tracking"]["wandb"]["tags"]
    assert "experiment-c" in generated["tracking"]["wandb"]["tags"]


def test_beit3_preprocessing_matches_vqav2_recipe() -> None:
    pipeline = build_image_transform(480, train=True, beit3_preprocessing=True)

    assert len(pipeline.transforms) == 3
    assert isinstance(pipeline.transforms[0], transforms.Resize)
    assert pipeline.transforms[0].size == (480, 480)
    assert pipeline.transforms[0].interpolation == transforms.InterpolationMode.BICUBIC
    assert isinstance(pipeline.transforms[1], transforms.ToTensor)
    assert isinstance(pipeline.transforms[2], transforms.Normalize)
    assert pipeline.transforms[2].mean == [0.5, 0.5, 0.5]
    assert pipeline.transforms[2].std == [0.5, 0.5, 0.5]


def test_beit3_tokenizer_local_only_requires_verified_cache(tmp_path, monkeypatch) -> None:
    cached = tmp_path / beit3.BEIT3_SENTENCEPIECE_FILENAME
    cached.write_bytes(b"tokenizer")
    monkeypatch.setattr(beit3, "_sha256", lambda _path: beit3.BEIT3_SENTENCEPIECE_SHA256)

    assert beit3.resolve_beit3_tokenizer(tmp_path, local_files_only=True) == cached
    with pytest.raises(FileNotFoundError, match="not cached"):
        beit3.resolve_beit3_tokenizer(tmp_path / "missing", local_files_only=True)


def test_beit3_full_stage_keeps_unused_mask_token_frozen(monkeypatch) -> None:
    class FakeConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeBackbone(nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.vision_embed = nn.Module()
            self.vision_embed.mask_token = nn.Parameter(torch.ones(1, 1, beit3.BEIT3_HIDDEN_SIZE))
            self.image_projection = nn.Linear(3, beit3.BEIT3_HIDDEN_SIZE)
            self.projection = nn.Linear(beit3.BEIT3_HIDDEN_SIZE, beit3.BEIT3_HIDDEN_SIZE)

        def forward(self, textual_tokens, visual_tokens, text_padding_position):
            _ = textual_tokens, text_padding_position
            pooled = self.image_projection(visual_tokens.mean(dim=(-1, -2))).unsqueeze(1)
            hidden = self.projection(pooled)
            return {"encoder_out": hidden}

    modules = {
        "torchscale": types.ModuleType("torchscale"),
        "torchscale.architecture": types.ModuleType("torchscale.architecture"),
        "torchscale.architecture.config": types.ModuleType("torchscale.architecture.config"),
        "torchscale.model": types.ModuleType("torchscale.model"),
        "torchscale.model.BEiT3": types.ModuleType("torchscale.model.BEiT3"),
    }
    modules["torchscale.architecture.config"].EncoderConfig = FakeConfig
    modules["torchscale.model.BEiT3"].BEiT3 = FakeBackbone
    monkeypatch.setitem(sys.modules, "torchscale", modules["torchscale"])
    monkeypatch.setitem(sys.modules, "torchscale.architecture", modules["torchscale.architecture"])
    monkeypatch.setitem(sys.modules, "torchscale.architecture.config", modules["torchscale.architecture.config"])
    monkeypatch.setitem(sys.modules, "torchscale.model", modules["torchscale.model"])
    monkeypatch.setitem(sys.modules, "torchscale.model.BEiT3", modules["torchscale.model.BEiT3"])

    model = build_model(
        {
            "name": "beit3_base",
            "image_size": 16,
            "gradient_checkpointing": False,
        },
        answer_vocab_size=5,
        load_pretrained=False,
    )
    model.set_finetune_stage("full")

    assert not model.beit3.vision_embed.mask_token.requires_grad
    assert model.beit3.projection.weight.requires_grad
    assert model.head[-1].weight.requires_grad
    logits = model(torch.rand(2, 3, 16, 16), torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4))
    assert logits.shape == (2, 5)
