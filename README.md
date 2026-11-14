# Edge Model Compression: Quantization, Pruning & Edge Deployment

**Course**: Edge Computing & Intelligent Systems (Fall 2026)  
**Assignment**: Assignment 03  
**Student**: Aimal Khan (`i211701`, Section `A`)  
**Institution**: FAST NUCES  
**Repository**: [github.com/aimal-khan/Edge-Computing-Assignment-03](https://github.com/aimal-khan/Edge-Computing-Assignment-03)  
**Deliverable Archive**: `i211701_A.zip`  

---

## Overview

Deploying deep neural networks to battery-powered or low-power embedded processors is fundamentally constrained by memory bandwidth. On modern edge platforms (such as ARM Cortex-A or Apple Silicon), fetching weights from external DRAM burns significantly more energy than computing the arithmetic operations themselves.

This repository implements, evaluates, and deploys two model compression strategies on CIFAR-10 using a customized ResNet-18 backbone:

1. **Integer Quantization (Question 1)**:
   - **Post-Training Static Quantization (PTQ)**: Fuses `Conv2d + BatchNorm2d + ReLU` operator chains, computes activation dynamic ranges over 500 calibration samples using PyTorch's `qnnpack` engine, and lowers numerical precision from 32-bit float to 8-bit integer.
   - **Quantization-Aware Training (QAT)**: Simulates rounding noise in the forward pass while propagating gradients through a Straight-Through Estimator (STE), recovering lost accuracy prior to final INT8 serialization.

2. **Network Pruning (Question 2)**:
   - **Iterative Unstructured Pruning**: Eliminates weight elements with small $L_1$ magnitudes over a three-stage schedule until reaching 50% sparsity, fine-tunes between steps, and permanently bakes the binary masks into the weights.
   - **Structured Channel Pruning**: Ranks convolutional filters by their $L_2$ norm, prunes 30% of internal residual channels, and **physically reconstructs** smaller `nn.Conv2d` and `nn.BatchNorm2d` layers to produce real CPU speedups without needing sparse matrix software libraries.

3. **Cross-Platform Edge Deployment**:
   - Compiles every model variant into ONNX graphs (opset 13) and benchmarks them on the CPU execution provider.

---

## Experimental Benchmark Summary

All benchmarks were evaluated on the full 10,000-image CIFAR-10 test partition on an Apple M1 processor (ARM64, 16 GB unified RAM):

| Model Variant | Strategy | Top-1 Accuracy | Single-Image CPU Latency | Checkpoint Size | Non-Zero Parameters | Sparsity | Storage Reduction | Latency Speedup |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ResNet-18** | FP32 Dense | 83.62% | 9.30 ms | 42.70 MB | 11,173,962 | 0.00% | 1.00x | 1.00x |
| **PTQ-INT8** | Static Quantization | 83.62% | 3.75 ms | 10.71 MB | 11,173,962 | 0.00% | **3.99x** | **2.48x** |
| **QAT-INT8** | Quantization-Aware Training | **83.99%** | **3.67 ms** | **10.71 MB** | 11,173,962 | 0.00% | **3.99x** | **2.53x** |
| **Unstructured Pruned** | Iterative $L_1$ Magnitude | 84.35% | 9.25 ms | 42.70 MB | 5,591,785 | **49.96%** | 1.00x | 1.01x |
| **Structured Pruned** | Filter $L_2$ + Rebuilt Graph | 83.52% | **7.45 ms** | **30.09 MB** | **7,864,834** | 0.00% | **1.42x** | **1.25x** |

### Key Takeaways
- **INT8 Quantization** delivered the cleanest edge speedup: cutting memory footprint by **75%** (42.7 MB down to 10.7 MB) and reducing latency by **60%** (9.30 ms to 3.67 ms) with zero accuracy degradation (+0.37% with QAT).
- **Unstructured sparsity** zeroes out 50% of weights but gives virtually no latency gain (9.25 ms vs 9.30 ms) on standard CPUs, because regular dense matrix libraries still multiply and fetch zeroes.
- **Structured channel pruning** rebuilds genuine smaller layers, cutting parameter counts by **29.61%** and producing real wall-clock latency drops (**7.45 ms**).

---

## Directory Structure

```
Edge-Computing-Assignment-03/
├── Assignment 3.pdf                  # Official course prompt and assignment rubric
├── README.md                         # Project documentation and execution instructions
├── requirements.txt                  # Python dependencies
├── run_all.py                        # Master pipeline script running all steps sequentially
│
├── common/
│   ├── dataset.py                    # CIFAR-10 data loaders & 500-sample calibration split
│   └── model_factory.py              # CIFAR-adapted ResNet-18 with quant stubs & fusion hooks
│
├── q1_quantization/
│   ├── train_baseline.py             # Baseline FP32 ResNet-18 training script
│   ├── ptq.py                        # Post-Training Static Quantization with calibration
│   ├── qat.py                        # Quantization-Aware Training with fake quant & STE
│   ├── inference.py                  # CLI inference utility for FP32, PTQ, and QAT models
│   └── benchmark.py                  # Profiling script generating results_q1.csv & predictions
│
├── q2_pruning/
│   ├── unstructured_pruning.py       # Iterative 3-stage L1 magnitude pruning & mask consolidation
│   ├── structured_pruning.py         # L2 filter pruning & physical layer tensor slicing
│   ├── inference.py                  # CLI inference utility for pruned models
│   └── benchmark.py                  # Profiling script generating results_q2.csv & predictions
│
├── deployment/
│   ├── export_onnx.py                # Serializes all 5 PyTorch models to ONNX graphs
│   └── onnx_inference.py             # ONNX Runtime CPU inference runner
│
├── models/                           # PyTorch checkpoints (.pth)
│   ├── resnet18_fp32.pth             # Baseline FP32 weights (42.7 MB)
│   ├── resnet18_ptq_int8.pth         # Static quantized INT8 weights (10.7 MB)
│   ├── resnet18_qat_int8.pth         # QAT fine-tuned INT8 weights (10.7 MB)
│   ├── resnet18_unstructured.pth     # 50% unstructured pruned weights (42.7 MB)
│   └── resnet18_structured.pth       # 30% channel pruned reconstructed model (30.1 MB)
│
├── onnx/                             # Exported ONNX graphs (.onnx)
│   ├── resnet18_fp32.onnx
│   ├── resnet18_ptq_int8.onnx
│   ├── resnet18_qat_int8.onnx
│   ├── resnet18_unstructured.onnx
│   └── resnet18_structured.onnx
│
├── results/                          # Benchmark metric logs & predictions
│   ├── results_q1.csv                # Accuracy, latency, and size for Q1 models
│   ├── results_q2.csv                # Accuracy, latency, and sparsity for Q2 models
│   ├── predictions_q1.csv            # Test sample predictions (truth vs predicted)
│   ├── predictions_q2.csv            # Test sample predictions for pruned variants
│   └── full_benchmark_comparison.csv # Unified comparison table across all 5 models
│
└── report/
    ├── report.md                     # In-depth technical report with formulations
    ├── Technical_Report_i211701_A.pdf# Formatted submission PDF with bold black headings
    ├── generate_plots.py             # Matplotlib script creating publication figures
    ├── generate_pdf.py               # ReportLab PDF compilation script
    └── plots/                        # Generated performance charts
        ├── accuracy_comparison.png
        ├── latency_comparison.png
        ├── size_comparison.png
        └── tradeoff_pareto.png
```

---

## Deliverables Checklist

| Rubric Requirement | Implementation Location | Verified Deliverable |
|:---|:---|:---:|
| Baseline ResNet-18 model & training script | `q1_quantization/train_baseline.py` | `models/resnet18_fp32.pth` (83.62%) |
| Post-Training Quantization (PTQ) & calibration | `q1_quantization/ptq.py` | `models/resnet18_ptq_int8.pth` (10.7 MB) |
| Quantization-Aware Training (QAT) & fine-tuning | `q1_quantization/qat.py` | `models/resnet18_qat_int8.pth` (83.99%) |
| Unstructured iterative pruning (50% target) | `q2_pruning/unstructured_pruning.py` | `models/resnet18_unstructured.pth` (49.96% sparsity) |
| Structured channel pruning (30% physical rebuild) | `q2_pruning/structured_pruning.py` | `models/resnet18_structured.pth` (29.61% params cut) |
| Unified inference CLI for all checkpoints | `q1_quantization/inference.py`, `q2_pruning/inference.py` | Tested on sample images |
| Benchmark logging & prediction CSVs | `q1_quantization/benchmark.py`, `q2_pruning/benchmark.py` | `results/*.csv` generated |
| ONNX graph export & CPU inference engine | `deployment/export_onnx.py`, `deployment/onnx_inference.py` | `onnx/*.onnx` running via ONNX Runtime |
| Technical report & publication plots | `report/report.md`, `report/generate_pdf.py` | `report/Technical_Report_i211701_A.pdf` |
| Master execution script | `run_all.py` | Single-command workflow |

---

## Setup and Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/aimal-khan/Edge-Computing-Assignment-03.git
cd Edge-Computing-Assignment-03

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> **Hardware Note**: For Apple Silicon (M1/M2/M3), PyTorch eager-mode quantization uses the `qnnpack` backend (`torch.backends.quantized.engine = 'qnnpack'`). The scripts configure this automatically based on platform detection.

---

### 2. Running the Complete Pipeline

To run the entire pipeline end-to-end (training, quantization, pruning, benchmarking, ONNX export, and plot generation):

```bash
python run_all.py
```

---

### 3. Step-by-Step Individual Commands

#### Step 1: Baseline Training
```bash
python q1_quantization/train_baseline.py --epochs 5 --batch-size 128 --lr 0.05
```

#### Step 2: Post-Training Quantization (PTQ)
```bash
python q1_quantization/ptq.py --baseline ./models/resnet18_fp32.pth --save-path ./models/resnet18_ptq_int8.pth --calib-samples 500
```

#### Step 3: Quantization-Aware Training (QAT)
```bash
python q1_quantization/qat.py --baseline ./models/resnet18_fp32.pth --save-path ./models/resnet18_qat_int8.pth --epochs 2 --lr 1e-4
```

#### Step 4: Quantization Benchmarking
```bash
python q1_quantization/benchmark.py --output ./results/results_q1.csv
```

#### Step 5: Unstructured Iterative Pruning
```bash
python q2_pruning/unstructured_pruning.py --baseline ./models/resnet18_fp32.pth --save-path ./models/resnet18_unstructured.pth --sparsity 0.50 --iterations 3 --epochs 2
```

#### Step 6: Structured Channel Pruning & Physical Reconstruction
```bash
python q2_pruning/structured_pruning.py --baseline ./models/resnet18_fp32.pth --save-path ./models/resnet18_structured.pth --ratio 0.30 --epochs 2
```

#### Step 7: Pruning Benchmarking
```bash
python q2_pruning/benchmark.py --output ./results/results_q2.csv
```

#### Step 8: ONNX Export and Profiling
```bash
# Export all models to ONNX
python deployment/export_onnx.py

# Benchmark single-image CPU inference via ONNX Runtime
python deployment/onnx_inference.py --model-path ./onnx/resnet18_structured.onnx --benchmark
```

#### Step 9: Interactive Model Inference CLI
```bash
# Test quantized model
python q1_quantization/inference.py --model qat

# Test pruned model
python q2_pruning/inference.py --model structured
```

---

## Citation & Academic Integrity

This project was developed for the **Edge Computing & Intelligent Systems** course at **FAST NUCES** (Fall 2026) by **Aimal Khan (i211701, Section A)**. All benchmarks were recorded on an Apple M1 system using Python 3.9, PyTorch 2.8, and ONNX Runtime 1.19.
