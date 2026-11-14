import os
import sys
import time
import subprocess
import csv

PYTHON_EXEC = sys.executable

def log_section(title: str):
    print("\n" + "=" * 80)
    print(f"  >>> {title.upper()} <<<")
    print("=" * 80 + "\n")

def run_command(cmd_list):
    print(f"[CMD] {' '.join(cmd_list)}")
    start_time = time.time()
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"[!] Command failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[DONE] Elapsed: {time.time() - start_time:.1f}s\n")

def consolidate_benchmarks():
    log_section("Consolidating Cross-Method Benchmarks")
    q1_csv = "./results/results_q1.csv"
    q2_csv = "./results/results_q2.csv"
    full_csv = "./results/full_benchmark_comparison.csv"

    rows = []
    if os.path.exists(q1_csv):
        with open(q1_csv, mode="r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "Technique": "Quantization" if "INT8" in r["Model"] else "Baseline",
                    "Model": r["Model"],
                    "Accuracy (%)": r.get("Accuracy (%)", ""),
                    "Latency (ms/img)": r.get("Latency per Image (ms)", ""),
                    "Model Size (MB)": r.get("Model Size (MB)", ""),
                    "Parameters": r.get("Parameters", "")
                })

    if os.path.exists(q2_csv):
        with open(q2_csv, mode="r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["Model"] == "Original FP32":
                    continue  # Already represented from Q1
                rows.append({
                    "Technique": "Pruning",
                    "Model": r["Model"],
                    "Accuracy (%)": r.get("Accuracy (%)", ""),
                    "Latency (ms/img)": r.get("Latency per Image (ms)", ""),
                    "Model Size (MB)": r.get("Model Size (MB)", ""),
                    "Parameters": r.get("Total Parameters", "")
                })

    with open(full_csv, mode="w", newline="") as f:
        fieldnames = ["Technique", "Model", "Accuracy (%)", "Latency (ms/img)", "Model Size (MB)", "Parameters"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] Unified benchmark summary written to: {full_csv}")
    print("\n" + "-" * 75)
    print(f"{'Technique':<14} | {'Model':<22} | {'Acc (%)':<8} | {'Lat (ms)':<9} | {'Size (MB)':<9}")
    print("-" * 75)
    for r in rows:
        print(f"{r['Technique']:<14} | {r['Model']:<22} | {r['Accuracy (%)']:<8} | {r['Latency (ms/img)']:<9} | {r['Model Size (MB)']:<9}")
    print("-" * 75 + "\n")

def main():
    total_start = time.time()
    log_section("Starting Complete Optimization Pipeline Execution")

    # Step 1: Baseline FP32 Training
    log_section("Step 1: Train Baseline ResNet-18 FP32 (CIFAR-10)")
    run_command([PYTHON_EXEC, "q1_quantization/train_baseline.py", "--epochs", "5", "--batch-size", "128"])

    # Step 2: Post-Training Static Quantization (PTQ)
    log_section("Step 2: Post-Training Static Quantization (PTQ)")
    run_command([PYTHON_EXEC, "q1_quantization/ptq.py", "--calib-samples", "500"])

    # Step 3: Quantization-Aware Training (QAT)
    log_section("Step 3: Quantization-Aware Training (QAT)")
    run_command([PYTHON_EXEC, "q1_quantization/qat.py", "--epochs", "2", "--lr", "1e-4"])

    # Step 4: Benchmark Quantization Models
    log_section("Step 4: Benchmark Q1 Quantization Artifacts")
    run_command([PYTHON_EXEC, "q1_quantization/benchmark.py"])

    # Step 5: Unstructured Pruning
    log_section("Step 5: Iterative Unstructured Pruning + Fine-tuning")
    run_command([PYTHON_EXEC, "q2_pruning/unstructured_pruning.py", "--sparsity", "0.50", "--iterations", "3", "--epochs", "2"])

    # Step 6: Structured Channel Pruning & Architecture Reconstruction
    log_section("Step 6: Structured Channel Pruning & Reconstruction")
    run_command([PYTHON_EXEC, "q2_pruning/structured_pruning.py", "--ratio", "0.30", "--epochs", "2"])

    # Step 7: Benchmark Pruned Models
    log_section("Step 7: Benchmark Q2 Pruning Artifacts")
    run_command([PYTHON_EXEC, "q2_pruning/benchmark.py"])

    # Step 8: Export All Models to ONNX
    log_section("Step 8: Export All 5 Models to ONNX")
    run_command([PYTHON_EXEC, "deployment/export_onnx.py"])

    # Step 9: Consolidate Benchmarks
    consolidate_benchmarks()

    # Step 10: Generate Visualization Charts for Technical Report
    if os.path.exists("report/generate_plots.py"):
        log_section("Step 10: Generating Visualization Figures")
        run_command([PYTHON_EXEC, "report/generate_plots.py"])

    total_elapsed = time.time() - total_start
    log_section(f"Pipeline Completed Successfully in {total_elapsed / 60.0:.2f} minutes")

if __name__ == "__main__":
    main()
