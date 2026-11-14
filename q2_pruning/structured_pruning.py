import os
import sys
import argparse
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import get_cifar10_loaders
from common.model_factory import (
    create_cifar_resnet18,
    count_parameters,
    get_model_size_mb
)


def evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    """Measure empirical classification accuracy across evaluation tensors."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def prune_and_reconstruct_channel(
    model: nn.Module,
    prune_ratio: float = 0.30
) -> Tuple[nn.Module, Dict[str, int]]:
    """Prune convolutional channels via L2-norm ranking and physically reconstruct dimensions.
    
    Identifies bottom-ranked feature maps within internal residual block convolutions,
    slicing both convolutional weight kernels and batch normalization moments.
    Reconstructs authentic lower-dimensional nn.Conv2d instances to guarantee physical
    execution without synthetic zero-mask dependency.
    """
    model.eval()
    channel_cfg = {}

    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    for layer_name in layer_names:
        layer = getattr(model, layer_name)
        for block_idx, block in enumerate(layer):
            conv1 = block.conv1
            bn1 = block.bn1
            conv2 = block.conv2

            in_c = conv1.in_channels
            out_c = conv1.out_channels
            k = max(1, int(out_c * (1.0 - prune_ratio)))
            key = f"{layer_name}.{block_idx}.conv1"
            channel_cfg[key] = k

            # 1. Compute L2-norm across output channels
            weights = conv1.weight.data
            norms = torch.norm(weights.view(out_c, -1), p=2, dim=1)
            _, keep_indices = torch.topk(norms, k, largest=True)
            keep_indices, _ = torch.sort(keep_indices)

            # 2. Physically instantiate reduced conv1
            new_conv1 = nn.Conv2d(
                in_channels=in_c,
                out_channels=k,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=(conv1.bias is not None)
            )
            new_conv1.weight.data = conv1.weight.data[keep_indices].clone()
            if conv1.bias is not None:
                new_conv1.bias.data = conv1.bias.data[keep_indices].clone()

            # 3. Physically instantiate reduced bn1
            new_bn1 = nn.BatchNorm2d(k)
            new_bn1.weight.data = bn1.weight.data[keep_indices].clone()
            new_bn1.bias.data = bn1.bias.data[keep_indices].clone()
            new_bn1.running_mean.data = bn1.running_mean.data[keep_indices].clone()
            new_bn1.running_var.data = bn1.running_var.data[keep_indices].clone()

            # 4. Physically instantiate reduced conv2 (adjusting input channel dimension)
            new_conv2 = nn.Conv2d(
                in_channels=k,
                out_channels=conv2.out_channels,
                kernel_size=conv2.kernel_size,
                stride=conv2.stride,
                padding=conv2.padding,
                bias=(conv2.bias is not None)
            )
            new_conv2.weight.data = conv2.weight.data[:, keep_indices].clone()
            if conv2.bias is not None:
                new_conv2.bias.data = conv2.bias.data.clone()

            # 5. Transplant reconstructed modules back into residual block
            block.conv1 = new_conv1
            block.bn1 = new_bn1
            block.conv2 = new_conv2

    return model, channel_cfg


def fine_tune_structured(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-3
):
    """Restore empirical generalization capability on the reconstructed graph."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Structured Fine-tune [{epoch}/{epochs}]", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc = evaluate(model, test_loader, device)
        print(f"[*] Recovery Epoch {epoch:02d} | Val Accuracy: {acc:.2f}%")
    return model


def run_structured_pruning(
    baseline_path: str = "./models/resnet18_fp32.pth",
    save_path: str = "./models/resnet18_structured.pth",
    prune_ratio: float = 0.30,
    fine_tune_epochs: int = 2,
    batch_size: int = 128
):
    """Execute end-to-end structured channel pruning, tensor reconstruction, and recovery."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Initializing Structured Channel Pruning on {device}...")

    # 1. Load baseline model
    model = create_cifar_resnet18(num_classes=10)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint absent: {baseline_path}. Train baseline first.")
    model.load_state_dict(torch.load(baseline_path, map_location="cpu"))

    total_orig, _ = count_parameters(model)
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    init_acc = evaluate(model.to(device), test_loader, device)
    print(f"[*] Baseline Model Parameters: {total_orig:,} | Validation Acc: {init_acc:.2f}%")

    # 2. Prune channels and reconstruct architecture
    print(f"[+] Pruning {prune_ratio * 100:.1f}% channels across internal block convolutions...")
    model_cpu = model.cpu()
    reconstructed_model, cfg = prune_and_reconstruct_channel(model_cpu, prune_ratio=prune_ratio)
    total_reconstructed, _ = count_parameters(reconstructed_model)
    param_reduction = (total_orig - total_reconstructed) / total_orig * 100.0

    print(f"[*] Reconstructed Architecture Parameters: {total_reconstructed:,}")
    print(f"[*] Physical Parameter Elimination: {param_reduction:.2f}% reduction")

    # Verify zero-mask independence (tensor validation)
    dummy = torch.randn(2, 3, 32, 32)
    _ = reconstructed_model(dummy)
    print("[✓] Tensor dimension verification passed without dimension mismatches.")

    # 3. Fine-tune reconstructed graph
    reconstructed_model = reconstructed_model.to(device)
    post_prune_acc = evaluate(reconstructed_model, test_loader, device)
    print(f"[*] Immediate Post-Prune Accuracy: {post_prune_acc:.2f}%")

    print(f"[+] Initiating fine-tuning recovery across {fine_tune_epochs} epochs...")
    fine_tune_structured(reconstructed_model, train_loader, test_loader, device, epochs=fine_tune_epochs, lr=1e-3)

    final_acc = evaluate(reconstructed_model, test_loader, device)
    print(f"[*] Final Recovered Accuracy: {final_acc:.2f}%")

    # 4. Save deployable reconstructed artifact
    reconstructed_cpu = reconstructed_model.cpu()
    torch.save(reconstructed_cpu, save_path)
    file_size = get_model_size_mb(save_path)
    print(f"[✓] Reconstructed structured model artifact persisted: {save_path} ({file_size:.2f} MB)")
    return reconstructed_cpu, final_acc, param_reduction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Channel Pruning for ResNet-18.")
    parser.add_argument("--baseline", type=str, default="./models/resnet18_fp32.pth")
    parser.add_argument("--save-path", type=str, default="./models/resnet18_structured.pth")
    parser.add_argument("--ratio", type=float, default=0.30, help="Channel pruning ratio (e.g. 0.30)")
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning recovery epochs")
    args = parser.parse_args()

    run_structured_pruning(
        baseline_path=args.baseline,
        save_path=args.save_path,
        prune_ratio=args.ratio,
        fine_tune_epochs=args.epochs
    )
