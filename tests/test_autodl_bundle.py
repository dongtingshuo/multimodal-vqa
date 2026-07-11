from __future__ import annotations

from pathlib import Path

from vqa_project.config import load_config
from vqa_project.training import resume_signature


def test_autodl_config_preserves_resume_signature() -> None:
    original = load_config("configs/kaggle_vilt.yaml")
    autodl = load_config("autodl/configs/vilt_resume_4090d.yaml")

    assert resume_signature(autodl) == resume_signature(original)
    assert autodl["data"]["num_workers"] == 8
    assert autodl["train"]["batch_size"] == 4
    assert autodl["train"]["gradient_accumulation_steps"] == 8
    assert autodl["tracking"]["wandb"]["enabled"] is False


def test_autodl_pipeline_requires_format_v3_resume() -> None:
    pipeline = Path("autodl/run_pipeline.sh").read_text(encoding="utf-8")
    setup = Path("autodl/setup.sh").read_text(encoding="utf-8")

    assert '--resume "${RUN_DIR}/latest.pt"' in pipeline
    assert "--epochs 10" in pipeline
    assert "--no-wandb" in pipeline
    assert "-m autodl.preflight" in setup
    assert "latest.pt" in setup and "best.pt" in setup
