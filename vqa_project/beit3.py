from __future__ import annotations

import hashlib
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

BEIT3_BASE_URL = "https://github.com/addf400/files/releases/download/beit3"
BEIT3_BASE_FILENAME = "beit3_base_patch16_224.pth"
BEIT3_BASE_SHA256 = "12ebc3d487f418d4f0f000d09719edfe7dcc89079da427f5580a7fb86ee3943a"
BEIT3_BASE_SIZE = 554_351_015
BEIT3_SENTENCEPIECE_FILENAME = "beit3.spm"
BEIT3_SENTENCEPIECE_SHA256 = "6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4"
BEIT3_SENTENCEPIECE_SIZE = 1_356_293
BEIT3_TORCHSCALE_COMMIT = "4d1e0e82e5adf86dd424f1463192635b73fc8efc"
BEIT3_HIDDEN_SIZE = 768
BEIT3_VOCAB_SIZE = 64_010


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, path: Path, expected_size: int, expected_sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == expected_size and _sha256(path) == expected_sha256:
        return path

    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 10):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "multimodal-vqa/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    temporary.unlink(missing_ok=True)
                    continue
                mode = "ab" if offset else "wb"
                with temporary.open(mode) as destination:
                    while chunk := response.read(8 * 1024 * 1024):
                        destination.write(chunk)
        except (OSError, urllib.error.URLError, TimeoutError):
            if attempt == 9:
                raise
            time.sleep(min(2**attempt, 30))
            continue

        if temporary.stat().st_size > expected_size:
            temporary.unlink()
            raise ValueError(f"Downloaded model asset exceeds its expected size: {path.name}")
        if temporary.stat().st_size < expected_size:
            if attempt == 9:
                raise OSError(f"Incomplete download for {path.name}: {temporary.stat().st_size}/{expected_size} bytes")
            time.sleep(min(2**attempt, 30))
            continue
        actual_sha256 = _sha256(temporary)
        if actual_sha256 != expected_sha256:
            temporary.unlink()
            raise ValueError(f"SHA-256 mismatch for {path.name}: {actual_sha256}")
        temporary.replace(path)
        return path
    raise OSError(f"Could not download {path.name} after repeated attempts")


def resolve_beit3_assets(cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    root = Path(cache_dir or os.environ.get("VQA_BEIT3_CACHE", Path.home() / ".cache" / "multimodal-vqa" / "beit3"))
    weights = _download_verified(
        f"{BEIT3_BASE_URL}/{BEIT3_BASE_FILENAME}",
        root / BEIT3_BASE_FILENAME,
        BEIT3_BASE_SIZE,
        BEIT3_BASE_SHA256,
    )
    tokenizer = resolve_beit3_tokenizer(root)
    return weights, tokenizer


def resolve_beit3_tokenizer(
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    root = Path(cache_dir or os.environ.get("VQA_BEIT3_CACHE", Path.home() / ".cache" / "multimodal-vqa" / "beit3"))
    tokenizer_path = root / BEIT3_SENTENCEPIECE_FILENAME
    if local_files_only:
        if not tokenizer_path.is_file() or _sha256(tokenizer_path) != BEIT3_SENTENCEPIECE_SHA256:
            raise FileNotFoundError(
                f"BEiT-3 tokenizer is not cached at {tokenizer_path}; provide a local beit3.spm file "
                "or launch without offline mode."
            )
        return tokenizer_path
    return _download_verified(
        f"{BEIT3_BASE_URL}/{BEIT3_SENTENCEPIECE_FILENAME}",
        tokenizer_path,
        BEIT3_SENTENCEPIECE_SIZE,
        BEIT3_SENTENCEPIECE_SHA256,
    )


def beit3_tokenizer(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.is_file() or _sha256(model_path) != BEIT3_SENTENCEPIECE_SHA256:
        raise ValueError(f"BEiT-3 SentencePiece model failed SHA-256 verification: {model_path}")
    try:
        from transformers import XLMRobertaTokenizer
    except ImportError as exc:
        raise RuntimeError("BEiT-3 requires transformers and the optional sentencepiece dependency.") from exc
    return XLMRobertaTokenizer(str(model_path))


def _canonical_key(name: str) -> str:
    while name.startswith(("module.", "model.")):
        name = name.split(".", 1)[1]
    return name.replace("._checkpoint_wrapped_module", "")


def _resize_vision_positions(state: dict[str, torch.Tensor], target_size: int) -> None:
    key = "beit3.encoder.embed_positions.A.weight"
    position = state.get(key)
    if position is None or position.shape[0] == target_size:
        return
    prefix_tokens = 3
    old_grid = math.isqrt(position.shape[0] - prefix_tokens)
    new_grid = math.isqrt(target_size - prefix_tokens)
    if old_grid * old_grid != position.shape[0] - prefix_tokens or new_grid * new_grid != target_size - prefix_tokens:
        raise ValueError(f"Unsupported BEiT-3 positional embedding shapes: {position.shape[0]} -> {target_size}")
    prefix = position[:prefix_tokens].float()
    patches = position[prefix_tokens:].float().reshape(old_grid, old_grid, -1).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(patches, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    state[key] = torch.cat(
        [prefix, resized.squeeze(0).permute(1, 2, 0).reshape(-1, position.shape[1])], dim=0
    )


class BEiT3VQAModel(nn.Module):
    permanently_frozen_parameter_names = {"beit3.vision_embed.mask_token"}

    def __init__(
        self,
        answer_vocab_size: int,
        image_size: int = 480,
        drop_path_rate: float = 0.1,
        gradient_checkpointing: bool = True,
        pretrained_model_path: str | None = None,
        cache_dir: str | None = None,
        load_pretrained: bool = True,
        mock_backbones: bool = False,
        mock_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        if mock_backbones:
            self.beit3 = nn.Sequential(nn.Flatten(), nn.Linear(3 * image_size * image_size, mock_hidden_size))
            self.pooler = nn.Identity()
            self.head = nn.Linear(mock_hidden_size, answer_vocab_size)
            self.pretrained_metadata = {"loaded": False, "mock": True}
            return

        try:
            from torchscale.architecture.config import EncoderConfig
            from torchscale.model.BEiT3 import BEiT3
        except ImportError as exc:
            raise RuntimeError("BEiT-3 requires the optional dependencies: torchscale, fairscale, and sentencepiece.") from exc

        args = EncoderConfig(
            img_size=int(image_size),
            patch_size=16,
            vocab_size=BEIT3_VOCAB_SIZE,
            multiway=True,
            layernorm_embedding=False,
            normalize_output=False,
            no_output_layer=True,
            drop_path_rate=float(drop_path_rate),
            encoder_embed_dim=BEIT3_HIDDEN_SIZE,
            encoder_attention_heads=12,
            encoder_ffn_embed_dim=BEIT3_HIDDEN_SIZE * 4,
            encoder_layers=12,
            checkpoint_activations=bool(gradient_checkpointing),
        )
        self.beit3 = BEiT3(args)
        self.pooler = _BEiT3Pooler(BEIT3_HIDDEN_SIZE)
        self.head = nn.Sequential(
            nn.Linear(BEIT3_HIDDEN_SIZE, BEIT3_HIDDEN_SIZE * 2),
            nn.LayerNorm(BEIT3_HIDDEN_SIZE * 2),
            nn.GELU(),
            nn.Linear(BEIT3_HIDDEN_SIZE * 2, int(answer_vocab_size)),
        )
        self._initialize_head()

        self.pretrained_metadata = {"loaded": False, "source": "checkpoint_or_random_initialization"}
        if load_pretrained:
            path = Path(pretrained_model_path) if pretrained_model_path else None
            if path is None:
                path, _ = resolve_beit3_assets(cache_dir)
            if not path.is_file() or _sha256(path) != BEIT3_BASE_SHA256:
                raise ValueError(f"BEiT-3 base checkpoint failed SHA-256 verification: {path}")
            self.pretrained_metadata = self.load_pretrained(path)

    def set_finetune_stage(self, stage: str, **kwargs) -> None:
        _ = kwargs
        if stage not in {"frozen", "full"}:
            raise ValueError("BEiT-3 supports only frozen or full-backbone fine-tuning stages.")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(stage == "full" and name not in self.permanently_frozen_parameter_names)
        if stage == "frozen":
            for parameter in (*self.pooler.parameters(), *self.head.parameters()):
                parameter.requires_grad_(True)
        self.finetune_stage = stage

    def _initialize_head(self) -> None:
        for module in (self.pooler, self.head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def load_pretrained(self, path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("module", payload))
        if not isinstance(state, dict):
            raise ValueError("BEiT-3 checkpoint does not contain a model state dictionary.")
        state = {str(name): value for name, value in state.items() if torch.is_tensor(value)}
        positional_key = "beit3.encoder.embed_positions.A.weight"
        target_state = self.state_dict()
        target_position = target_state[positional_key].shape[0]
        _resize_vision_positions(state, target_position)

        target_by_canonical = {_canonical_key(name): name for name in target_state if name.startswith("beit3.")}
        remapped: dict[str, torch.Tensor] = {}
        for source_name, tensor in state.items():
            target_name = target_by_canonical.get(_canonical_key(source_name))
            if target_name is not None and tensor.shape == target_state[target_name].shape:
                remapped[target_name] = tensor
        total = sum(value.numel() for name, value in target_state.items() if name.startswith("beit3."))
        loaded = sum(target_state[name].numel() for name in remapped)
        coverage = loaded / max(total, 1)
        loaded_layers = {
            int(name.split(".layers.", 1)[1].split(".", 1)[0])
            for name in remapped
            if ".layers." in name
        }
        if coverage < 0.98 or len(loaded_layers) != 12:
            raise RuntimeError(
                "BEiT-3 pretrained weights did not load completely: "
                f"backbone_numel_coverage={coverage:.4%}, loaded_layers={len(loaded_layers)}/12"
            )
        self.load_state_dict(remapped, strict=False)
        print(
            f"BEiT-3 pretrained weights verified: backbone coverage={coverage:.2%}, "
            f"layers={len(loaded_layers)}/12, sha256={BEIT3_BASE_SHA256}"
        )
        return {
            "loaded": True,
            "source": BEIT3_BASE_FILENAME,
            "sha256": BEIT3_BASE_SHA256,
            "backbone_numel_coverage": coverage,
            "loaded_layers": len(loaded_layers),
        }

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.pretrained_metadata.get("mock"):
            return self.head(self.beit3(images))
        _ = kwargs
        outputs = self.beit3(
            textual_tokens=input_ids,
            visual_tokens=images,
            text_padding_position=attention_mask == 0,
        )
        return self.head(self.pooler(outputs["encoder_out"]))


class _BEiT3Pooler(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(self.norm(hidden_states[:, 0, :])))
