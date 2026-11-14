import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import get_cifar10_loaders
from common.model_factory import (
    create_cifar_resnet18,
    configure_quant_engine,
    get_model_size_mb
)


def evaluate_model(model: nn.Module, test_loader) -> float:
    """Assess classification accuracy on CIFAR-10 evaluation partition."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating QAT", leave=False):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def run_qat(
    baseline_path: str = "./models/resnet18_fp32.pth",
    save_path: str = "./models/resnet18_qat_int8.pth",
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-4,
    max_batches: int = 50
):
    """Execute Quantization-Aware Training (QAT) fine-tuning pipeline."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    backend = configure_quant_engine()
    print(f"[+] Initializing QAT with quantization backend: {backend}")

    # 1. Instantiate and load pretrained FP32 checkpoint
    model = create_cifar_resnet18(num_classes=10)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint absent: {baseline_path}. Train baseline first.")
    model.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    model.train()

    # 2. Fuse modules for QAT
    print("[+] Fusing conv, bn, and relu modules with QAT semantics...")
    model.fuse_model(is_qat=True)

    # 3. Configure QAT qconfig
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    # 4. Insert fake-quantization operators
    print("[+] Embedding fake-quantization modules via torch.ao.quantization.prepare_qat...")
    torch.ao.quantization.prepare_qat(model, inplace=True)

    # 5. Fine-tuning QAT pipeline
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    print(f"[+] Launching QAT fine-tuning for {epochs} epoch(s) (cap: {max_batches} batches)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch [{epoch}/{epochs}]", total=min(len(train_loader), max_batches), leave=False)

        for b_idx, (inputs, targets) in enumerate(pbar):
            if b_idx >= max_batches:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Freeze batch norm statistics for stabilization before conversion
        print("[*] Freezing batch normalization statistics for stable integer conversion...")
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        epoch_loss = running_loss / (max_batches * batch_size)
        print(f"[*] QAT Epoch {epoch:02d} complete | Fine-tune Loss: {epoch_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"[✓] QAT fine-tuning finished in {elapsed:.1f}s.")

    # 6. Convert simulated QAT model to native INT8 representation
    print("[+] Converting simulated QAT graph into physical INT8 representation...")
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)

    # 7. Evaluate converted INT8 model
    acc = evaluate_model(quantized_model, test_loader)
    print(f"[*] Converted QAT-INT8 Test Accuracy: {acc:.2f}%")

    # 8. Persist deployable TorchScript artifact
    print(f"[+] Exporting optimized INT8 model checkpoint to {save_path}...")
    scripted_model = torch.jit.script(quantized_model)
    scripted_model.save(save_path)

    fp32_size = get_model_size_mb(baseline_path)
    qat_size = get_model_size_mb(save_path)
    compression_ratio = fp32_size / qat_size if qat_size > 0 else 0.0

    print(f"\n[✓] QAT complete!")
    print(f"[✓] FP32 checkpoint size : {fp32_size:.2f} MB")
    print(f"[✓] QAT-INT8 model size  : {qat_size:.2f} MB")
    print(f"[✓] Compression ratio    : {compression_ratio:.2f}x reduction")
    print(f"[✓] Persisted artifact   : {save_path}")
    return quantized_model, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization-Aware Training (QAT) for ResNet-18.")
    parser.add_argument("--baseline", type=str, default="./models/resnet18_fp32.pth", help="Source FP32 checkpoint")
    parser.add_argument("--save-path", type=str, default="./models/resnet18_qat_int8.pth", help="Target INT8 path")
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for QAT fine-tuning")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum training batches for fine-tuning")
    args = parser.parse_args()

    run_qat(
        baseline_path=args.baseline,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_batches=args.max_batches
    )
