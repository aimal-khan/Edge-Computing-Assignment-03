import os
import sys
import argparse
import time
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import CIFAR10_CLASSES, obtain_transforms, get_cifar10_loaders


def create_onnx_session(onnx_path: str) -> ort.InferenceSession:
    """Instantiate an ONNX Runtime InferenceSession bound to CPUExecutionProvider."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX artifact nonexistent at {onnx_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    return session


def run_onnx_single(session: ort.InferenceSession, image_path: str = ""):
    """Execute ONNX CPU inference on a single image and record latency statistics."""
    _, test_tf = obtain_transforms()

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = test_tf(img).unsqueeze(0).numpy()
    else:
        print("[!] Generating synthetic 1x3x32x32 numpy array.")
        tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)

    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: tensor})

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probs)[::-1][:3]

    avg_lat = sum(latencies) / len(latencies)
    print("\n" + "=" * 50)
    print(f"ONNX CPU Inference (Latency: {avg_lat:.2f} ms):")
    for rank, idx in enumerate(top_indices):
        c_name = CIFAR10_CLASSES[idx] if idx < len(CIFAR10_CLASSES) else f"Class_{idx}"
        print(f"  {rank+1}. {c_name:<10}: {probs[idx] * 100:.2f}%")
    print("=" * 50 + "\n")
    return top_indices[0], avg_lat


def benchmark_onnx_on_cifar10(session: ort.InferenceSession, max_samples: int = 1000):
    """Evaluate accuracy and throughput across CIFAR-10 evaluation partition in ONNX Runtime."""
    _, test_loader = get_cifar10_loaders(batch_size=64)
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    latencies = []

    for inputs, targets in test_loader:
        batch_np = inputs.numpy()
        targets_np = targets.numpy()

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: batch_np})
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt / batch_np.shape[0])

        preds = np.argmax(outputs[0], axis=1)
        correct += int((preds == targets_np).sum())
        total += batch_np.shape[0]

        if total >= max_samples:
            break

    acc = (correct / total) * 100.0
    mean_lat = sum(latencies) / len(latencies)
    print(f"[*] Evaluated {total} samples | ONNX Accuracy: {acc:.2f}% | Latency: {mean_lat:.3f} ms/image")
    return acc, mean_lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX CPU Inference and Benchmarking Engine.")
    parser.add_argument("--model-path", type=str, default="./onnx/resnet18_fp32.onnx", help="Path to .onnx file")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on CIFAR-10 subset")
    args = parser.parse_args()

    sess = create_onnx_session(args.model_path)
    if args.benchmark:
        benchmark_onnx_on_cifar10(sess)
    else:
        run_onnx_single(sess, args.image)
