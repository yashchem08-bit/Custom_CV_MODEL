import torch
import torch.nn as nn
from typing import List


class SqueezeExcitation(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation).
    Adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV3 / EfficientNet inspired Inverted Residual Bottleneck:
    1. Pointwise Conv 1x1 (Expansion)
    2. Depthwise Conv 3x3 (Spatial filtering)
    3. Squeeze-and-Excitation (Channel attention)
    4. Pointwise Conv 1x1 (Linear Projection)
    5. Skip-connection when stride == 1 and in_channels == out_channels
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 4, use_se: bool = True):
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)

        layers: List[nn.Module] = []

        # 1. Expansion (Pointwise)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True)
            ])

        # 2. Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True)
        ])

        # 3. Squeeze and Excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))

        # 4. Linear Projection (No non-linearity to preserve linear manifold)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class CustomVisionBackbone(nn.Module):
    """
    High-efficiency multi-stage feature extractor.
    Optimized for high throughput on both CPUs and GPUs.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Stem: Initial downsampling [B, 3, 224, 224] -> [B, 32, 112, 112]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True)
        )

        # Stage 1: [B, 32, 112, 112] -> [B, 32, 112, 112]
        self.stage1 = InvertedResidualBlock(32, 32, stride=1, expand_ratio=1, use_se=False)

        # Stage 2: [B, 32, 112, 112] -> [B, 64, 56, 56]
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=4, use_se=True),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4, use_se=True)
        )

        # Stage 3: [B, 64, 56, 56] -> [B, 128, 28, 28]
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(64, 128, stride=2, expand_ratio=4, use_se=True),
            InvertedResidualBlock(128, 128, stride=1, expand_ratio=4, use_se=True)
        )

        # Stage 4: [B, 128, 28, 28] -> [B, 256, 14, 14]
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(128, 256, stride=2, expand_ratio=6, use_se=True),
            InvertedResidualBlock(256, 256, stride=1, expand_ratio=6, use_se=True)
        )

        # Final projection before head: [B, 256, 14, 14] -> [B, 512, 14, 14]
        self.conv_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_head(x)
        return x


class CustomVisionModel(nn.Module):
    """
    Complete end-to-end Vision Model with Backbone + Pooling + Classification Head.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone = CustomVisionBackbone(in_channels=in_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.gap(features)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits


if __name__ == "__main__":
    # Smoke test: instantiate and check parameter count & forward pass
    model = CustomVisionModel(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"Model initialized successfully!")
    print(f"Trainable Parameters: {total_params / 1e6:.2f}M")
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
