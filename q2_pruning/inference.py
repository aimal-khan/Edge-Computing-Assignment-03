import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dataset import CIFAR10_CLASSES, obtain_transforms
from common.model_factory import create_cifar_resnet18


def load_pruned_model_for_inference(model_type: str, checkpoint_path: str = None):
    """Restore targeted pruned architecture for active inference evaluation."""
    if model_type.lower() in ["original", "fp32"]:
        path = checkpoint_path or "./models/resnet18_fp32.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing original checkpoint: {path}")
        model = create_cifar_resnet18(num_classes=10)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    elif model_type.lower() in ["unstructured", "unstructured-pruned"]:
        path = checkpoint_path or "./models/resnet18_unstructured.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing unstructured checkpoint: {path}")
        model = create_cifar_resnet18(num_classes=10)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    elif model_type.lower() in ["structured", "structured-pruned"]:
        path = checkpoint_path or "./models/resnet18_structured.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing structured checkpoint: {path}")
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model

    else:
        raise ValueError(f"Unrecognized pruned model type: {model_type}. Options: original, unstructured, structured")


def run_pruned_inference(model, image_path: str = ""):
    """Classify input image and evaluate real-time CPU latency over 50 iterations."""
    _, test_tf = obtain_transforms()

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = test_tf(img).unsqueeze(0)
    else:
        print("[!] Generating synthesized 32x32 tensor for evaluation pass.")
        tensor = torch.randn(1, 3, 32, 32)

    # Warmup invocation
    with torch.no_grad():
        _ = model(tensor)

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
    print(f"Pruned Model Inference (Mean Latency: {avg_latency:.2f} ms):")
    for i in range(top_prob.size(0)):
        idx = top_idx[i].item()
        c_name = CIFAR10_CLASSES[idx] if idx < len(CIFAR10_CLASSES) else f"Class_{idx}"
        print(f"  {i+1}. {c_name:<10}: {top_prob[i].item() * 100:.2f}%")
    print("=" * 50 + "\n")
    return top_idx[0].item(), avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference CLI for Pruned ResNet-18 variants.")
    parser.add_argument("--model", type=str, default="structured", choices=["original", "unstructured", "structured"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default="")
    args = parser.parse_args()

    m = load_pruned_model_for_inference(args.model, args.checkpoint)
    run_pruned_inference(m, args.image)
