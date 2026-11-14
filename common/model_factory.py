import os
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models.quantization as qmodels


def create_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Instantiate a ResNet-18 variant parameterized specifically for 32x32 CIFAR-10 images.
    
    Substitutes aggressive 7x7 spatial downsampling and max pooling with a non-strided
    3x3 convolutional receptive field, preserving spatial granularity and facilitating
    eager quantization through integrated QuantStub and DeQuantStub blocks.
    """
    model = qmodels.resnet18(weights=None)
    # Adapt stem for 32x32 spatial dimensions
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Tally aggregate parameters alongside structurally non-zero coefficients."""
    total_params = 0
    non_zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        non_zero_params += int((param != 0).sum().item())
    return total_params, non_zero_params


def get_model_size_mb(filepath: str) -> float:
    """Ascertain persisted checkpoint physical volume in megabytes."""
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / (1024.0 * 1024.0)


def configure_quant_engine():
    """Designate hardware-appropriate quantization instruction engine."""
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
    return torch.backends.quantized.engine
