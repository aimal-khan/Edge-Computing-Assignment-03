import os
import sys
import argparse
import torch
import onnx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.model_factory import create_cifar_resnet18, get_model_size_mb, configure_quant_engine


def export_single_model(model, output_path: str, dummy_input: torch.Tensor):
    """Serialize target PyTorch graph into standardized ONNX format, verifying schema."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    print(f"[+] Serializing model graph to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )

    # Validate ONNX schema integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mb = get_model_size_mb(output_path)
    print(f"[✓] ONNX validation succeeded: {output_path} ({size_mb:.2f} MB)")


def run_all_exports(
    models_dir: str = "./models",
    onnx_dir: str = "./onnx"
):
    """Systematically export all 5 model variations (FP32, PTQ, QAT, Unstructured, Structured)."""
    os.makedirs(onnx_dir, exist_ok=True)
    configure_quant_engine()
    dummy_input = torch.randn(1, 3, 32, 32)

    export_configs = [
        ("FP32 Baseline", os.path.join(models_dir, "resnet18_fp32.pth"), os.path.join(onnx_dir, "resnet18_fp32.onnx"), "fp32"),
        ("PTQ-INT8", os.path.join(models_dir, "resnet18_ptq_int8.pth"), os.path.join(onnx_dir, "resnet18_ptq_int8.onnx"), "jit"),
        ("QAT-INT8", os.path.join(models_dir, "resnet18_qat_int8.pth"), os.path.join(onnx_dir, "resnet18_qat_int8.onnx"), "jit"),
        ("Unstructured Pruned", os.path.join(models_dir, "resnet18_unstructured.pth"), os.path.join(onnx_dir, "resnet18_unstructured.onnx"), "fp32"),
        ("Structured Pruned", os.path.join(models_dir, "resnet18_structured.pth"), os.path.join(onnx_dir, "resnet18_structured.onnx"), "full_model"),
    ]

    print("\n" + "=" * 70)
    print("ONNX EXPORT PIPELINE FOR EDGE DEPLOYMENT")
    print("=" * 70)

    for name, in_path, out_path, mode in export_configs:
        if not os.path.exists(in_path):
            print(f"[!] Warning: Checkpoint {in_path} absent. Skipping {name}.")
            continue

        print(f"\n[+] Preparing {name} from {in_path}...")
        try:
            if mode == "fp32":
                model = create_cifar_resnet18(num_classes=10)
                model.load_state_dict(torch.load(in_path, map_location="cpu", weights_only=True))
            elif mode == "jit":
                model = torch.jit.load(in_path, map_location="cpu")
            elif mode == "full_model":
                model = torch.load(in_path, map_location="cpu", weights_only=False)

            export_single_model(model, out_path, dummy_input)
        except Exception as e:
            print(f"[!] Error exporting {name}: {e}")

    print("\n[✓] All accessible models processed for ONNX deployment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX format.")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--onnx-dir", type=str, default="./onnx")
    args = parser.parse_args()

    run_all_exports(models_dir=args.models_dir, onnx_dir=args.onnx_dir)
