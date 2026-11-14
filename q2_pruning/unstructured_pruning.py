import os
import sys
import argparse
import time
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
    """Measure top-1 validation accuracy across target hardware tensors."""
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


def calculate_sparsity(model: nn.Module) -> Tuple[float, int, int]:
    """Inspect parameter tensors to quantify global structural zero-ratio."""
    total_elements = 0
    zero_elements = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total_elements += param.numel()
            zero_elements += int((param == 0).sum().item())
    sparsity = (zero_elements / total_elements) * 100.0 if total_elements > 0 else 0.0
    return sparsity, zero_elements, total_elements


def fine_tune(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-3
):
    """Execute compensatory fine-tuning to recover accuracy under zero-masked weights."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fine-tuning [{epoch}/{epochs}]", leave=False)
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


def run_unstructured_pruning(
    baseline_path: str = "./models/resnet18_fp32.pth",
    save_path: str = "./models/resnet18_unstructured.pth",
    target_sparsity: float = 0.50,
    iterations: int = 3,
    fine_tune_epochs: int = 2,
    batch_size: int = 128
):
    """Implement iterative magnitude-based unstructured pruning across Conv and Linear layers."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Initializing Unstructured Pruning Pipeline on {device}...")

    # 1. Instantiate and load baseline
    model = create_cifar_resnet18(num_classes=10)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint absent: {baseline_path}. Train baseline first.")
    model.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    model = model.to(device)

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    init_acc = evaluate(model, test_loader, device)
    print(f"[*] Baseline Unpruned Accuracy: {init_acc:.2f}%")

    # Collect prunable layers (Conv2d and Linear)
    prunable_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable_modules.append((name, module))
    print(f"[+] Identified {len(prunable_modules)} candidate convolutional & linear modules.")

    # 2. Iterative pruning loop
    # Formulate per-iteration incremental sparsity amounts
    # If target is 50% across 3 rounds: 1 - (1 - target)**(1/iters)
    per_iter_rate = 1.0 - (1.0 - target_sparsity) ** (1.0 / iterations)

    for it in range(1, iterations + 1):
        print(f"\n[+] Pruning Iteration [{it}/{iterations}] (Incremental Rate: {per_iter_rate:.3f})...")
        for name, module in prunable_modules:
            prune.l1_unstructured(module, name="weight", amount=per_iter_rate)

        sp, zeros, total = calculate_sparsity(model)
        acc_post_prune = evaluate(model, test_loader, device)
        print(f"[*] Post-prune Sparsity: {sp:.2f}% | Intermediate Accuracy: {acc_post_prune:.2f}%")

        if it < iterations:
            # Intermediate brief fine-tune
            fine_tune(model, train_loader, test_loader, device, epochs=1, lr=5e-4)

    # 3. Final compensatory fine-tuning
    print(f"\n[+] Executing terminal fine-tuning recovery across {fine_tune_epochs} epochs...")
    fine_tune(model, train_loader, test_loader, device, epochs=fine_tune_epochs, lr=1e-3)

    # 4. Make pruning permanent by removing reparameterization hooks
    print("[+] Removing forward reparameterization hooks to solidify weight masks permanently...")
    for name, module in prunable_modules:
        prune.remove(module, "weight")

    # 5. Validate final metrics
    final_sp, _, _ = calculate_sparsity(model)
    final_acc = evaluate(model, test_loader, device)
    print(f"[*] Solidified Permanent Sparsity: {final_sp:.2f}%")
    print(f"[*] Recovered Validation Accuracy: {final_acc:.2f}%")

    # 6. Save permanent pruned model checkpoint
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), save_path)
    file_size = get_model_size_mb(save_path)
    print(f"[✓] Permanent unstructured pruned artifact persisted: {save_path} ({file_size:.2f} MB)")
    return model_cpu, final_acc, final_sp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unstructured Pruning for ResNet-18.")
    parser.add_argument("--baseline", type=str, default="./models/resnet18_fp32.pth")
    parser.add_argument("--save-path", type=str, default="./models/resnet18_unstructured.pth")
    parser.add_argument("--sparsity", type=float, default=0.50, help="Target weight sparsity (e.g. 0.50)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of pruning iterations")
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning recovery epochs")
    args = parser.parse_args()

    run_unstructured_pruning(
        baseline_path=args.baseline,
        save_path=args.save_path,
        target_sparsity=args.sparsity,
        iterations=args.iterations,
        fine_tune_epochs=args.epochs
    )
