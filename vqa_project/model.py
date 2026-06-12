from __future__ import annotations

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


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.image_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
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

        mask = text_attention_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
        text_sum = (text_tokens * mask).sum(dim=1)
        text_denominator = mask.sum(dim=1).clamp_min(1.0)
        pooled_text = text_sum / text_denominator
        pooled_image = image_tokens.mean(dim=1)
        return torch.cat([pooled_image, pooled_text], dim=-1)


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
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_cnn)
        self.text_encoder = load_text_model(text_model_name)
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
        image_tokens = self.image_encoder(images)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_outputs.last_hidden_state

        image_tokens = self.image_projection(image_tokens)
        text_tokens = self.text_projection(text_tokens)
        fused = self.fusion(image_tokens, text_tokens, attention_mask)
        return self.classifier(fused)
