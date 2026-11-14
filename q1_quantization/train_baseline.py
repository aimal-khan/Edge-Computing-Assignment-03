import os
import sys
import argparse
import time
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Ensure workspace root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import get_cifar10_loaders
from common.model_factory import create_cifar_resnet18, count_parameters, get_model_size_mb


def evaluate(model: nn.Module, test_loader, device: torch.device) -> Tuple[float, float]:
    """Scrutinize validation tensors to measure top-1 empirical classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def train_baseline(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.05,
    save_path: str = "./models/resnet18_fp32.pth",
    device_name: str = "auto"
):
    """Execute training regime for baseline ResNet-18 architecture on CIFAR-10."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[+] Initializing baseline training on hardware accelerator: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    model = create_cifar_resnet18(num_classes=10).to(device)

    total_params, _ = count_parameters(model)
    print(f"[+] Instantiated ResNet-18 parameter volume: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        acc, test_loss = evaluate(model, test_loader, device)
        print(f"[*] Epoch {epoch:02d} complete | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            # Persist optimal empirical parameters
            torch.save(model.state_dict(), save_path)

    elapsed = time.time() - start_time
    file_size_mb = get_model_size_mb(save_path)
    print(f"\n[✓] Baseline FP32 training complete in {elapsed:.1f}s.")
    print(f"[✓] Checkpoint saved: {save_path} ({file_size_mb:.2f} MB)")
    print(f"[✓] Best Validation Accuracy: {best_acc:.2f}%")
    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline FP32 ResNet-18 model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for loaders")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    parser.add_argument("--save-path", type=str, default="./models/resnet18_fp32.pth", help="Checkpoint target path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'mps', 'auto')")
    args = parser.parse_args()

    train_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        device_name=args.device
    )
