"""
CNN model for SVHN digit recognition.

Implements an enhanced architecture with:
- Spatial Transformer Network (STN) for geometric invariance
- Inception-ResNet blocks for multi-scale feature extraction
- Squeeze-and-Excitation (SE) blocks for channel attention
- Multi-task learning with 6 output heads

Based on CNN_IMPLEMENTATION.md and existing SVHN_CNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class STN(nn.Module):
    """
    Spatial Transformer Network for learning invariance to spatial transformations.
    """

    def __init__(self, input_channels: int = 3):
        super().__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 6 affine transformation parameters
        )

        # Initialize to identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        # Predict transformation parameters
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Apply transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x, theta


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Squeeze: global average pooling
        y = self.squeeze(x).view(batch, channels)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(batch, channels, 1, 1)

        # Scale: multiply input by attention weights
        return x * y.expand_as(x)


class InceptionResidualBlock(nn.Module):
    """
    Inception-ResNet block combining multi-scale features with skip connections.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Ensure dimensions match for skip connection
        self.match_dimensions = (stride != 1 or in_channels != out_channels)
        if self.match_dimensions:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        # Inception branches
        # Branch 1: 1x1 conv (with pooling if stride > 1)
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.MaxPool2d(3, stride=stride, padding=1),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            )

        # Branch 2: 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3,
                     stride=stride, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 -> two 3x3 convs (simulates 5x5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3,
                     stride=stride, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 maxpool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # SE block for channel attention
        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = x

        # Inception branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate branches
        out = torch.cat([b1, b2, b3, b4], dim=1)

        # Apply SE block
        out = self.se(out)

        # Add skip connection
        if self.match_dimensions:
            identity = self.shortcut(identity)

        out += identity
        out = F.relu(out)

        return out


class MultiTaskHead(nn.Module):
    """
    Multi-task learning head with 6 outputs.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()

        # Shared representation
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Length prediction head
        self.length_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5)  # 0-4 digits
        )

        # Digit classification heads
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 11)  # 0-9 + blank
            ) for _ in range(4)
        ])

        # Has digits classifier (binary)
        self.presence_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, features):
        # Shared processing
        shared = self.shared_fc(features)

        # Task-specific outputs
        outputs = {
            'num': self.length_head(shared),
            'dig1': self.digit_heads[0](shared),
            'dig2': self.digit_heads[1](shared),
            'dig3': self.digit_heads[2](shared),
            'dig4': self.digit_heads[3](shared),
            'nC': self.presence_head(shared)
        }

        return outputs


class EnhancedSVHNCNN(nn.Module):
    """
    Enhanced CNN model for SVHN digit recognition.

    Architecture:
    1. Spatial Transformer Network
    2. Inception-ResNet feature extraction
    3. Multi-task prediction heads
    """

    def __init__(self, input_channels: int = 3, use_stn: bool = True):
        super().__init__()

        self.use_stn = use_stn

        # Stage 1: Spatial Transformer
        if use_stn:
            self.stn = STN(input_channels=input_channels)

        # Stage 2: Feature extraction with Inception-ResNet blocks
        self.layer1 = InceptionResidualBlock(input_channels, 64, stride=1)    # 32x32 -> 32x32
        self.layer2 = InceptionResidualBlock(64, 128, stride=2)   # 32x32 -> 16x16
        self.layer3 = InceptionResidualBlock(128, 256, stride=2)  # 16x16 -> 8x8
        self.layer4 = InceptionResidualBlock(256, 512, stride=2)  # 8x8 -> 4x4

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Stage 3: Multi-task heads
        self.multi_task_head = MultiTaskHead(feature_dim=512)

    def forward(self, x, return_theta: bool = False):
        theta = None

        # Spatial transformation
        if self.use_stn:
            x, theta = self.stn(x)

        # Feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Multi-task prediction
        outputs = self.multi_task_head(x)

        if return_theta and self.use_stn:
            return outputs, theta

        return outputs


def create_model(model_type: str = 'enhanced', input_channels: int = 3, use_stn: bool = True):
    """
    Factory function to create different model variants.

    Args:
        model_type: Type of model ('enhanced', 'basic')
        input_channels: Number of input channels
        use_stn: Whether to use Spatial Transformer Network

    Returns:
        Model instance
    """
    if model_type == 'enhanced':
        return EnhancedSVHNCNN(input_channels=input_channels, use_stn=use_stn)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test the model
    model = create_model('enhanced', use_stn=True)
    print(f"Model created successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    outputs, theta = model(x, return_theta=True)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    if theta is not None:
        print(f"  theta: {theta.shape}")
