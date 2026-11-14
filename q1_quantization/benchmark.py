import os
import sys
import argparse
import time
import csv
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import get_cifar10_loaders, CIFAR10_CLASSES
from common.model_factory import (
    create_cifar_resnet18,
    configure_quant_engine,
    get_model_size_mb,
    count_parameters
)


def evaluate_benchmark(model, test_loader, num_samples: int = None):
    """Derive rigorous empirical accuracy alongside latency figures over evaluation samples."""
    model.eval()
    correct = 0
    total = 0
    latencies = []
    sample_preds = []

    # Warmup invocation
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Benchmarking", leave=False):
            t0 = time.perf_counter()
            outputs = model(inputs)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt / inputs.size(0))

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if len(sample_preds) < 100:
                for b in range(inputs.size(0)):
                    if len(sample_preds) < 100:
                        sample_preds.append((
                            targets[b].item(),
                            predicted[b].item(),
                            CIFAR10_CLASSES[targets[b].item()],
                            CIFAR10_CLASSES[predicted[b].item()]
                        ))

            if num_samples is not None and total >= num_samples:
                break

    acc = 100.0 * correct / total
    avg_img_lat = sum(latencies) / len(latencies)
    return acc, avg_img_lat, sample_preds


def run_benchmark_q1(
    fp32_path: str = "./models/resnet18_fp32.pth",
    ptq_path: str = "./models/resnet18_ptq_int8.pth",
    qat_path: str = "./models/resnet18_qat_int8.pth",
    results_csv: str = "./results/results_q1.csv",
    predictions_csv: str = "./results/predictions_q1.csv"
):
    """Systematically evaluate FP32, PTQ, and QAT artifacts, logging empirical metrics."""
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    configure_quant_engine()
    _, test_loader = get_cifar10_loaders(batch_size=64)

    models_to_test = [
        ("FP32 Baseline", fp32_path, "fp32"),
        ("PTQ-INT8", ptq_path, "jit"),
        ("QAT-INT8", qat_path, "jit"),
    ]

    results = []
    all_predictions = {}
    base_size = get_model_size_mb(fp32_path)

    print("\n" + "=" * 70)
    print("INTEGER QUANTIZATION BENCHMARK PIPELINE")
    print("=" * 70)

    for name, path, mode in models_to_test:
        if not os.path.exists(path):
            print(f"[!] Warning: Artifact {path} absent. Skipping {name}.")
            continue

        size_mb = get_model_size_mb(path)
        comp_ratio = base_size / size_mb if size_mb > 0 else 1.0

        if mode == "fp32":
            model = create_cifar_resnet18(num_classes=10)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            total_params, _ = count_parameters(model)
        else:
            model = torch.jit.load(path, map_location="cpu")
            total_params = 11173962  # Standard ResNet-18 parameter equivalent

        print(f"\n[+] Benchmarking {name} ({size_mb:.2f} MB)...")
        acc, lat, preds = evaluate_benchmark(model, test_loader)
        all_predictions[name] = preds

        print(f"    - Accuracy           : {acc:.2f}%")
        print(f"    - Latency per Image  : {lat:.3f} ms")
        print(f"    - Disk Size          : {size_mb:.2f} MB")
        print(f"    - Compression Ratio  : {comp_ratio:.2f}x")

        results.append({
            "Model": name,
            "Accuracy (%)": f"{acc:.2f}",
            "Latency per Image (ms)": f"{lat:.3f}",
            "Model Size (MB)": f"{size_mb:.2f}",
            "Parameters": total_params,
            "Compression Ratio": f"{comp_ratio:.2f}x"
        })

    # Persist summary CSV
    with open(results_csv, mode="w", newline="") as f:
        fieldnames = ["Model", "Accuracy (%)", "Latency per Image (ms)", "Model Size (MB)", "Parameters", "Compression Ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Benchmarking metrics persisted: {results_csv}")

    # Persist prediction logs
    if all_predictions:
        with open(predictions_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sample_Index", "True_Label", "Class_Name"] + [f"{m}_Prediction" for m in all_predictions.keys()])
            first_model = list(all_predictions.keys())[0]
            n_samples = len(all_predictions[first_model])
            for i in range(n_samples):
                true_lbl, _, true_cls, _ = all_predictions[first_model][i]
                row = [i, true_lbl, true_cls]
                for m in all_predictions.keys():
                    row.append(all_predictions[m][i][3])
                writer.writerow(row)
        print(f"[✓] Prediction comparison log persisted: {predictions_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Q1 Quantization Models.")
    parser.add_argument("--fp32", type=str, default="./models/resnet18_fp32.pth")
    parser.add_argument("--ptq", type=str, default="./models/resnet18_ptq_int8.pth")
    parser.add_argument("--qat", type=str, default="./models/resnet18_qat_int8.pth")
    parser.add_argument("--output", type=str, default="./results/results_q1.csv")
    args = parser.parse_args()

    run_benchmark_q1(
        fp32_path=args.fp32,
        ptq_path=args.ptq,
        qat_path=args.qat,
        results_csv=args.output
    )
