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
    get_model_size_mb,
    count_parameters
)


def evaluate_benchmark(model, test_loader, num_samples: int = None):
    """Ascertain accuracy, per-instance latency, and specimen predictions."""
    model.eval()
    correct = 0
    total = 0
    latencies = []
    sample_preds = []

    # Warmup
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Benchmarking Pruned", leave=False):
            t0 = time.perf_counter()
            outputs = model(inputs)
            dt = (time.perf_counter() - t0) * 1000.0
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


def run_benchmark_q2(
    original_path: str = "./models/resnet18_fp32.pth",
    unstructured_path: str = "./models/resnet18_unstructured.pth",
    structured_path: str = "./models/resnet18_structured.pth",
    results_csv: str = "./results/results_q2.csv",
    predictions_csv: str = "./results/predictions_q2.csv"
):
    """Systematically evaluate original vs pruned variants, documenting comparative telemetry."""
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    _, test_loader = get_cifar10_loaders(batch_size=64)

    models_to_test = [
        ("Original FP32", original_path, "state_dict"),
        ("Unstructured Pruned", unstructured_path, "state_dict"),
        ("Structured Pruned", structured_path, "full_model"),
    ]

    results = []
    all_predictions = {}
    base_params = 11173962

    print("\n" + "=" * 70)
    print("NEURAL NETWORK PRUNING BENCHMARK PIPELINE")
    print("=" * 70)

    for name, path, mode in models_to_test:
        if not os.path.exists(path):
            print(f"[!] Warning: Artifact {path} absent. Skipping {name}.")
            continue

        size_mb = get_model_size_mb(path)
        if mode == "state_dict":
            model = create_cifar_resnet18(num_classes=10)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            model = torch.load(path, map_location="cpu", weights_only=False)

        total_p, non_zero_p = count_parameters(model)
        sparsity = (1.0 - (non_zero_p / total_p)) * 100.0 if total_p > 0 else 0.0
        param_reduction = ((base_params - total_p) / base_params) * 100.0

        print(f"\n[+] Evaluating {name} ({size_mb:.2f} MB)...")
        acc, lat, preds = evaluate_benchmark(model, test_loader)
        all_predictions[name] = preds

        print(f"    - Accuracy            : {acc:.2f}%")
        print(f"    - Latency per Image   : {lat:.3f} ms")
        print(f"    - Total Parameters    : {total_p:,}")
        print(f"    - Non-Zero Parameters : {non_zero_p:,}")
        print(f"    - Weight Sparsity     : {sparsity:.2f}%")
        print(f"    - Param Reduction     : {param_reduction:.2f}%")

        results.append({
            "Model": name,
            "Accuracy (%)": f"{acc:.2f}",
            "Latency per Image (ms)": f"{lat:.3f}",
            "Total Parameters": total_p,
            "Non-Zero Parameters": non_zero_p,
            "Sparsity (%)": f"{sparsity:.2f}",
            "Model Size (MB)": f"{size_mb:.2f}",
            "Param Reduction (%)": f"{param_reduction:.2f}"
        })

    # Persist summary CSV
    with open(results_csv, mode="w", newline="") as f:
        fieldnames = ["Model", "Accuracy (%)", "Latency per Image (ms)", "Total Parameters", "Non-Zero Parameters", "Sparsity (%)", "Model Size (MB)", "Param Reduction (%)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Pruning benchmark metrics persisted: {results_csv}")

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
        print(f"[✓] Prediction log persisted: {predictions_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Q2 Pruned Models.")
    parser.add_argument("--original", type=str, default="./models/resnet18_fp32.pth")
    parser.add_argument("--unstructured", type=str, default="./models/resnet18_unstructured.pth")
    parser.add_argument("--structured", type=str, default="./models/resnet18_structured.pth")
    parser.add_argument("--output", type=str, default="./results/results_q2.csv")
    args = parser.parse_args()

    run_benchmark_q2(
        original_path=args.original,
        unstructured_path=args.unstructured,
        structured_path=args.structured,
        results_csv=args.output
    )
