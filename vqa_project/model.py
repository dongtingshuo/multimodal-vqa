from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .hf import load_text_model


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.out_channels = 2048

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        batch, channels, height, width = feature_map.shape
        return feature_map.view(batch, channels, height * width).transpose(1, 2)


class TinyImageEncoder(nn.Module):
    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        batch, channels, height, width = feature_map.shape
        return feature_map.view(batch, channels, height * width).transpose(1, 2)


class TinyTextModel(nn.Module):
    def __init__(self, hidden_size: int = 32, vocab_size: int = 4096) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        token_ids = input_ids.clamp_min(0).remainder(self.embedding.num_embeddings)
        return SimpleNamespace(last_hidden_state=self.norm(self.embedding(token_ids)))


def _build_image_encoder(pretrained_cnn: bool, mock_backbones: bool, mock_hidden_size: int) -> nn.Module:
    if mock_backbones:
        return TinyImageEncoder(out_channels=mock_hidden_size)
    return ImageEncoder(pretrained=pretrained_cnn)


def _build_text_encoder(text_model_name: str, mock_backbones: bool, mock_hidden_size: int) -> nn.Module:
    if mock_backbones:
        return TinyTextModel(hidden_size=mock_hidden_size)
    return load_text_model(text_model_name)


def _masked_mean(tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    token_sum = (tokens * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return token_sum / denominator


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _set_batch_norm_eval(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_padding_mask = text_attention_mask == 0
        attended_text, _ = self.text_to_image(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        attended_image, _ = self.image_to_text(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )

        text_tokens = self.text_norm(text_tokens + self.dropout(attended_text))
        image_tokens = self.image_norm(image_tokens + self.dropout(attended_image))

        pooled_text = _masked_mean(text_tokens, text_attention_mask)
        pooled_image = image_tokens.mean(dim=1)
        return torch.cat([pooled_image, pooled_text], dim=-1)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.scorer(tokens).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)


class StrongCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.image_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_pool = AttentionPool(hidden_dim)
        self.image_pool = AttentionPool(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_padding_mask = text_attention_mask == 0
        attended_text, _ = self.text_to_image(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        attended_image, _ = self.image_to_text(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )

        text_gate = torch.sigmoid(self.text_gate(torch.cat([text_tokens, attended_text], dim=-1)))
        image_gate = torch.sigmoid(self.image_gate(torch.cat([image_tokens, attended_image], dim=-1)))
        text_tokens = self.text_norm(text_tokens + self.dropout(text_gate * attended_text))
        image_tokens = self.image_norm(image_tokens + self.dropout(image_gate * attended_image))

        pooled_text = self.text_pool(text_tokens, text_attention_mask)
        pooled_image = self.image_pool(image_tokens)
        return torch.cat(
            [
                pooled_image,
                pooled_text,
                pooled_image * pooled_text,
                torch.abs(pooled_image - pooled_text),
            ],
            dim=-1,
        )


class VQAModel(nn.Module):
    def __init__(
        self,
        answer_vocab_size: int,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        freeze_backbones: bool = True,
        pretrained_cnn: bool = True,
        mock_backbones: bool = False,
        mock_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.image_encoder = _build_image_encoder(pretrained_cnn, mock_backbones, mock_hidden_size)
        self.text_encoder = _build_text_encoder(text_model_name, mock_backbones, mock_hidden_size)
        text_hidden_size = self.text_encoder.config.hidden_size

        self.image_projection = nn.Linear(self.image_encoder.out_channels, hidden_dim)
        self.text_projection = nn.Linear(text_hidden_size, hidden_dim)
        self.fusion = CrossAttentionFusion(hidden_dim, num_attention_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, answer_vocab_size),
        )
        self.backbones_frozen = False
        self.finetune_stage = "full"
        self._trainable_image_modules: list[nn.Module] = []
        self._trainable_text_modules: list[nn.Module] = []

        if freeze_backbones:
            self.freeze_backbones()

    def freeze_backbones(self) -> None:
        self.backbones_frozen = True
        self.finetune_stage = "frozen"
        self._trainable_image_modules = []
        self._trainable_text_modules = []
        _set_requires_grad(self.image_encoder, False)
        _set_requires_grad(self.text_encoder, False)

    def set_finetune_stage(
        self,
        stage: str,
        image_blocks: int = 1,
        text_layers: int = 2,
    ) -> None:
        if stage not in {"frozen", "partial", "full"}:
            raise ValueError(f"Unsupported fine-tune stage: {stage}")

        _set_requires_grad(self.image_encoder, False)
        _set_requires_grad(self.text_encoder, False)
        self._trainable_image_modules = []
        self._trainable_text_modules = []

        if stage == "partial":
            if isinstance(self.image_encoder, ImageEncoder):
                residual_blocks = list(self.image_encoder.features.children())[-4:]
                self._trainable_image_modules = residual_blocks[-max(1, min(image_blocks, 4)) :]
            else:
                self._trainable_image_modules = [self.image_encoder]

            transformer = getattr(self.text_encoder, "transformer", None)
            layers = list(getattr(transformer, "layer", []))
            if layers:
                self._trainable_text_modules = layers[-max(1, min(text_layers, len(layers))) :]
            else:
                self._trainable_text_modules = [self.text_encoder]

            for module in self._trainable_image_modules + self._trainable_text_modules:
                _set_requires_grad(module, True)
        elif stage == "full":
            _set_requires_grad(self.image_encoder, True)
            _set_requires_grad(self.text_encoder, True)

        self.finetune_stage = stage
        self.backbones_frozen = stage == "frozen"
        if self.training:
            self._apply_backbone_train_modes()

    def _apply_backbone_train_modes(self) -> None:
        if self.finetune_stage == "full":
            return
        self.image_encoder.eval()
        self.text_encoder.eval()
        if self.finetune_stage == "partial":
            for module in self._trainable_image_modules:
                module.train()
                _set_batch_norm_eval(module)
            for module in self._trainable_text_modules:
                module.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._apply_backbone_train_modes()
        return self

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self.image_encoder(images)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_outputs.last_hidden_state

        image_tokens = self.image_projection(image_tokens)
        text_tokens = self.text_projection(text_tokens)
        fused = self.fusion(image_tokens, text_tokens, attention_mask)
        return self.classifier(fused)


class StrongCrossAttentionVQAModel(VQAModel):
    def __init__(
        self,
        answer_vocab_size: int,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        freeze_backbones: bool = True,
        pretrained_cnn: bool = True,
        mock_backbones: bool = False,
        mock_hidden_size: int = 32,
    ) -> None:
        super().__init__(
            answer_vocab_size=answer_vocab_size,
            text_model_name=text_model_name,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            freeze_backbones=False,
            pretrained_cnn=pretrained_cnn,
            mock_backbones=mock_backbones,
            mock_hidden_size=mock_hidden_size,
        )
        self.fusion = StrongCrossAttentionFusion(hidden_dim, num_attention_heads, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, answer_vocab_size),
        )
        if freeze_backbones:
            self.freeze_backbones()


class BaselineConcatVQAModel(nn.Module):
    def __init__(
        self,
        answer_vocab_size: int,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        freeze_backbones: bool = True,
        pretrained_cnn: bool = True,
        mock_backbones: bool = False,
        mock_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        _ = num_attention_heads
        self.image_encoder = _build_image_encoder(pretrained_cnn, mock_backbones, mock_hidden_size)
        self.text_encoder = _build_text_encoder(text_model_name, mock_backbones, mock_hidden_size)
        text_hidden_size = self.text_encoder.config.hidden_size
        self.image_projection = nn.Linear(self.image_encoder.out_channels, hidden_dim)
        self.text_projection = nn.Linear(text_hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, answer_vocab_size),
        )
        self.backbones_frozen = False
        if freeze_backbones:
            self.freeze_backbones()

    def freeze_backbones(self) -> None:
        self.backbones_frozen = True
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbones_frozen:
            self.image_encoder.eval()
            self.text_encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self.image_projection(self.image_encoder(images)).mean(dim=1)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_projection(text_outputs.last_hidden_state)
        pooled_text = _masked_mean(text_tokens, attention_mask)
        return self.classifier(torch.cat([image_tokens, pooled_text], dim=-1))


class TextOnlyVQAModel(nn.Module):
    def __init__(
        self,
        answer_vocab_size: int,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        freeze_backbones: bool = True,
        pretrained_cnn: bool = True,
        mock_backbones: bool = False,
        mock_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        _ = num_attention_heads, pretrained_cnn
        self.text_encoder = _build_text_encoder(text_model_name, mock_backbones, mock_hidden_size)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, answer_vocab_size),
        )
        self.backbones_frozen = False
        if freeze_backbones:
            self.freeze_backbones()

    def freeze_backbones(self) -> None:
        self.backbones_frozen = True
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbones_frozen:
            self.text_encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = images
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_projection(text_outputs.last_hidden_state)
        return self.classifier(_masked_mean(text_tokens, attention_mask))


class ImageOnlyVQAModel(nn.Module):
    def __init__(
        self,
        answer_vocab_size: int,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        freeze_backbones: bool = True,
        pretrained_cnn: bool = True,
        mock_backbones: bool = False,
        mock_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        _ = text_model_name, num_attention_heads
        self.image_encoder = _build_image_encoder(pretrained_cnn, mock_backbones, mock_hidden_size)
        self.image_projection = nn.Linear(self.image_encoder.out_channels, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, answer_vocab_size),
        )
        self.backbones_frozen = False
        if freeze_backbones:
            self.freeze_backbones()

    def freeze_backbones(self) -> None:
        self.backbones_frozen = True
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbones_frozen:
            self.image_encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = input_ids, attention_mask
        image_tokens = self.image_projection(self.image_encoder(images)).mean(dim=1)
        return self.classifier(image_tokens)


MODEL_REGISTRY = {
    "cross_attention": VQAModel,
    "strong_cross_attention": StrongCrossAttentionVQAModel,
    "baseline_concat": BaselineConcatVQAModel,
    "text_only": TextOnlyVQAModel,
    "image_only": ImageOnlyVQAModel,
}


def build_model(config: dict, answer_vocab_size: int | None = None) -> nn.Module:
    model_cfg = dict(config.get("model", config))
    resolved_answer_vocab_size = answer_vocab_size or model_cfg.pop("answer_vocab_size", None)
    if resolved_answer_vocab_size is None:
        data_cfg = config.get("data", {})
        resolved_answer_vocab_size = data_cfg.get("answer_vocab_size")
    if resolved_answer_vocab_size is None:
        raise ValueError("answer_vocab_size must be provided in config['model'] or as an argument.")

    model_name = model_cfg.pop("name", "cross_attention")
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model name '{model_name}'. Available models: {available}") from exc
    return model_class(answer_vocab_size=int(resolved_answer_vocab_size), **model_cfg)
