"""
Character Query Transformer for SVHN digit detection and recognition.

This module implements a DETR-style transformer that uses learnable character
queries to simultaneously detect and classify digits in house number images.

Architecture:
- CNN backbone (ResNet50) for feature extraction
- 2D positional encoding for spatial features
- Transformer encoder for processing visual features
- Transformer decoder with learnable character queries
- Multi-task heads: classification, bounding boxes, confidence

Based on CQT_IMPLEMENTATION.md specifications.
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PositionEmbedding2D(nn.Module):
    """
    2D sinusoidal position embedding for spatial features.
    Similar to DETR's approach.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns:
            pos: Position encoding of shape [B, 2*num_pos_feats, H, W]
        """
        B, C, H, W = x.shape

        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)

        # Normalize to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W

        # Create dimension indices for sinusoidal encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Position embeddings for x and y
        pos_x = x_embed[:, None] / dim_t  # [W, num_pos_feats]
        pos_y = y_embed[:, None] / dim_t  # [H, num_pos_feats]

        # Apply sin/cos
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)  # [W, num_pos_feats]
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)  # [H, num_pos_feats]

        # Create 2D positional encoding
        # pos_y: [H, num_pos_feats] -> [num_pos_feats, H, 1] -> [num_pos_feats, H, W]
        # pos_x: [W, num_pos_feats] -> [num_pos_feats, 1, W] -> [num_pos_feats, H, W]
        pos_y = pos_y.permute(1, 0).unsqueeze(-1).repeat(1, 1, W)  # [num_pos_feats, H, W]
        pos_x = pos_x.permute(1, 0).unsqueeze(1).repeat(1, H, 1)   # [num_pos_feats, H, W]

        # Concatenate to get full positional encoding
        pos = torch.cat([pos_y, pos_x], dim=0)  # [2*num_pos_feats, H, W]

        # Add batch dimension
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2*num_pos_feats, H, W]

        return pos


class MLP(nn.Module):
    """Multi-layer perceptron for output heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatureExtractor(nn.Module):
    """
    CNN backbone for feature extraction.
    Uses ResNet50 with pretrained ImageNet weights.
    """

    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True,
                 feature_dim: int = 256):
        super().__init__()

        if backbone == 'resnet50':
            # Load pretrained ResNet50
            resnet = models.resnet50(pretrained=pretrained)

            # Remove final pooling and fc layers
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

            # ResNet50 outputs 2048 channels
            backbone_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection to feature_dim
        self.conv_proj = nn.Conv2d(backbone_out_channels, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Feature maps [B, feature_dim, H', W']
        """
        features = self.backbone(x)
        features = self.conv_proj(features)
        return features


class CharacterQueryTransformer(nn.Module):
    """
    Character Query Transformer for SVHN digit detection and recognition.

    Uses DETR-style architecture with learnable character queries.
    """

    def __init__(
        self,
        num_queries: int = 6,
        num_classes: int = 11,  # 0-9 + empty
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        backbone: str = 'resnet50',
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.d_model = d_model

        # Feature extraction
        self.backbone = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained_backbone,
            feature_dim=d_model
        )

        # Positional encoding
        self.pos_encoder = PositionEmbedding2D(num_pos_feats=d_model // 2)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Learnable query embeddings (positional queries)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Output heads
        # Classification head: predict digit class (0-9) or empty (10)
        self.class_head = MLP(d_model, d_model, num_classes, 3)

        # Bounding box head: predict [cx, cy, w, h] normalized to [0, 1]
        self.bbox_head = MLP(d_model, d_model, 4, 3)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize transformer parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Dictionary with:
                - 'class_logits': [B, num_queries, num_classes]
                - 'bbox_coords': [B, num_queries, 4]
        """
        B = images.shape[0]

        # Extract features
        features = self.backbone(images)  # [B, d_model, H', W']

        # Add positional encoding
        pos_encoding = self.pos_encoder(features)  # [B, d_model, H', W']

        # Flatten spatial dimensions
        # [B, d_model, H', W'] -> [B, d_model, H'*W'] -> [H'*W', B, d_model]
        B, C, H, W = features.shape
        features_flat = features.flatten(2).permute(2, 0, 1)  # [H'*W', B, d_model]
        pos_flat = pos_encoding.flatten(2).permute(2, 0, 1)  # [H'*W', B, d_model]

        # Add positional encoding to features
        features_encoded = features_flat + pos_flat

        # Transformer encoder
        memory = self.transformer_encoder(features_encoded)  # [H'*W', B, d_model]

        # Learnable queries
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, d_model]

        # Transformer decoder
        decoder_output = self.transformer_decoder(queries, memory)  # [num_queries, B, d_model]

        # Permute to batch first: [num_queries, B, d_model] -> [B, num_queries, d_model]
        decoder_output = decoder_output.permute(1, 0, 2)

        # Output heads
        class_logits = self.class_head(decoder_output)  # [B, num_queries, num_classes]
        bbox_coords = self.bbox_head(decoder_output).sigmoid()  # [B, num_queries, 4]

        return {
            'class_logits': class_logits,
            'bbox_coords': bbox_coords,
        }


def create_cqt_model(
    num_queries: int = 6,
    num_classes: int = 11,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    backbone: str = 'resnet50',
    pretrained_backbone: bool = True,
) -> CharacterQueryTransformer:
    """
    Factory function to create a Character Query Transformer model.

    Args:
        num_queries: Maximum number of digits to detect (default: 6)
        num_classes: Number of classes including empty (default: 11)
        d_model: Model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 3)
        num_decoder_layers: Number of decoder layers (default: 3)
        dim_feedforward: Feedforward dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        backbone: Backbone architecture (default: 'resnet50')
        pretrained_backbone: Use pretrained backbone (default: True)

    Returns:
        CharacterQueryTransformer model
    """
    model = CharacterQueryTransformer(
        num_queries=num_queries,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
    )

    return model
