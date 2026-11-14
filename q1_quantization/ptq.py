import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import get_calibration_loader, get_cifar10_loaders
from common.model_factory import (
    create_cifar_resnet18,
    configure_quant_engine,
    get_model_size_mb
)


def calibrate_model(model: nn.Module, calib_loader, num_batches: int = None):
    """Channel representative activation distributions across calibration tensors."""
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(calib_loader, desc="Calibrating PTQ")):
            model(images)
            if num_batches is not None and (idx + 1) >= num_batches:
                break


def evaluate_quantized(model: nn.Module, test_loader) -> float:
    """Measure empirical classification accuracy on CPU quantized tensors."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating PTQ", leave=False):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def run_ptq(
    baseline_path: str = "./models/resnet18_fp32.pth",
    save_path: str = "./models/resnet18_ptq_int8.pth",
    calib_samples: int = 500
):
    """Execute Post-Training Static Quantization (PTQ) workflow on ResNet-18."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    backend = configure_quant_engine()
    print(f"[+] Initializing PTQ with quantization instruction backend: {backend}")

    # 1. Instantiate and populate baseline FP32 weights
    float_model = create_cifar_resnet18(num_classes=10)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint absent: {baseline_path}. Train baseline first.")
    float_model.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    float_model.eval()

    # 2. Layer fusion (Conv + BN + ReLU)
    print("[+] Applying graph fusion across compatible convolutional and batch norm layers...")
    float_model.fuse_model(is_qat=False)

    # 3. Quantization configuration
    qconfig = torch.ao.quantization.get_default_qconfig(backend)
    float_model.qconfig = qconfig

    # 4. Prepare observers
    print("[+] Inserting calibration observers via torch.ao.quantization.prepare...")
    torch.ao.quantization.prepare(float_model, inplace=True)

    # 5. Calibration pipeline
    print(f"[+] Gathering empirical activation statistics on {calib_samples} uncorrupted training slices...")
    calib_loader = get_calibration_loader(subset_size=calib_samples, batch_size=32)
    calibrate_model(float_model, calib_loader)

    # 6. Conversion to INT8
    print("[+] Converting calibrated FP32 graph into native INT8 representation...")
    quantized_model = torch.ao.quantization.convert(float_model, inplace=False)

    # 7. Validation evaluation
    _, test_loader = get_cifar10_loaders(batch_size=128)
    acc = evaluate_quantized(quantized_model, test_loader)
    print(f"[*] Post-Training Quantized Model Test Accuracy: {acc:.2f}%")

    # 8. Deployable serialization using TorchScript
    print("[+] Persisting deployable TorchScript quantized artifact...")
    scripted_model = torch.jit.script(quantized_model)
    scripted_model.save(save_path)

    fp32_size = get_model_size_mb(baseline_path)
    int8_size = get_model_size_mb(save_path)
    compression_ratio = fp32_size / int8_size if int8_size > 0 else 0.0

    print(f"\n[✓] PTQ complete!")
    print(f"[✓] FP32 checkpoint size : {fp32_size:.2f} MB")
    print(f"[✓] INT8 checkpoint size : {int8_size:.2f} MB")
    print(f"[✓] Compression ratio    : {compression_ratio:.2f}x reduction")
    print(f"[✓] Persisted artifact   : {save_path}")
    return quantized_model, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-Training Static Quantization (PTQ) for ResNet-18.")
    parser.add_argument("--baseline", type=str, default="./models/resnet18_fp32.pth", help="Source FP32 checkpoint")
    parser.add_argument("--save-path", type=str, default="./models/resnet18_ptq_int8.pth", help="Target INT8 path")
    parser.add_argument("--calib-samples", type=int, default=500, help="Number of calibration samples")
    args = parser.parse_args()

    run_ptq(
        baseline_path=args.baseline,
        save_path=args.save_path,
        calib_samples=args.calib_samples
    )
