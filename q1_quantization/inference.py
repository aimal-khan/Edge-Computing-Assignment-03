import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import CIFAR10_CLASSES, obtain_transforms
from common.model_factory import create_cifar_resnet18, configure_quant_engine


def load_model_for_inference(model_type: str, checkpoint_path: str = None):
    """Instantiate and prime target neural graph for active inference."""
    configure_quant_engine()

    if model_type.lower() == "fp32":
        path = checkpoint_path or "./models/resnet18_fp32.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing FP32 checkpoint: {path}")
        model = create_cifar_resnet18(num_classes=10)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model, "cpu"

    elif model_type.lower() in ["ptq", "ptq-int8"]:
        path = checkpoint_path or "./models/resnet18_ptq_int8.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing PTQ-INT8 checkpoint: {path}")
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        return model, "cpu"

    elif model_type.lower() in ["qat", "qat-int8"]:
        path = checkpoint_path or "./models/resnet18_qat_int8.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing QAT-INT8 checkpoint: {path}")
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        return model, "cpu"

    else:
        raise ValueError(f"Unrecognized model variant: {model_type}. Options: fp32, ptq, qat")


def run_single_inference(model, image_path: str, device: str = "cpu"):
    """Execute classification and compute latency over an individual image specimen."""
    _, test_tf = obtain_transforms()

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = test_tf(img).unsqueeze(0).to(device)
    else:
        print("[!] Image path nonexistent or unprovided; generating synthesized input tensor.")
        tensor = torch.randn(1, 3, 32, 32).to(device)

    # Warmup invocation
    with torch.no_grad():
        _ = model(tensor)

    # Latency measurement
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            logits = model(tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    probs = F.softmax(logits, dim=1)[0]
    top_prob, top_idx = torch.topk(probs, 3)

    avg_latency = sum(latencies) / len(latencies)
    print("\n" + "=" * 50)
    print(f"Prediction Summary (Lat: {avg_latency:.2f} ms):")
    for i in range(top_prob.size(0)):
        idx = top_idx[i].item()
        c_name = CIFAR10_CLASSES[idx] if idx < len(CIFAR10_CLASSES) else f"Class_{idx}"
        print(f"  {i+1}. {c_name:<10}: {top_prob[i].item() * 100:.2f}%")
    print("=" * 50 + "\n")
    return top_idx[0].item(), avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Inference CLI for FP32, PTQ, and QAT models.")
    parser.add_argument("--model", type=str, default="fp32", choices=["fp32", "ptq", "qat"], help="Model variant")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    args = parser.parse_args()

    model, dev = load_model_for_inference(args.model, args.checkpoint)
    run_single_inference(model, args.image, device=dev)
