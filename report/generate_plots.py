import os
import csv
import matplotlib.pyplot as plt

def generate_report_figures(
    csv_path: str = "./results/full_benchmark_comparison.csv",
    output_dir: str = "./report/plots"
):
    """Render publication-grade analytical plots with an elegant, professional palette."""
    if not os.path.exists(csv_path):
        print(f"[!] Benchmark summary absent: {csv_path}. Skipping plot generation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    models = []
    accuracies = []
    latencies = []
    sizes = []

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row["Model"])
            accuracies.append(float(row["Accuracy (%)"]))
            latencies.append(float(row["Latency (ms/img)"]))
            sizes.append(float(row["Model Size (MB)"]))

    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    # Professional IEEE/Nature inspired palette:
    # Baseline: Slate, PTQ: Deep Navy, QAT: Vivid Cobalt, Unstructured: Steel Blue, Structured: Teal
    colors = ["#475569", "#1E3A8A", "#2563EB", "#0284C7", "#0D9488"]

    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    bars = ax.bar(models, accuracies, color=colors, width=0.48, edgecolor="#0F172A", linewidth=0.7, zorder=3)
    ax.set_title("CIFAR-10 Test Accuracy across Compression Methods", fontsize=11.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=9.5, fontweight="semibold", color="#334155")
    ax.set_ylim(80, 86)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.12, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax.tick_params(axis="x", labelrotation=12, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

    # 2. Latency Comparison
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    bars = ax.bar(models, latencies, color=colors, width=0.48, edgecolor="#0F172A", linewidth=0.7, zorder=3)
    ax.set_title("Single-Image CPU Inference Latency (Apple M1, batch=64)", fontsize=11.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_ylabel("Latency (ms / sample)", fontsize=9.5, fontweight="semibold", color="#334155")
    ax.set_ylim(0, max(latencies) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.18, f"{h:.2f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax.tick_params(axis="x", labelrotation=12, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"))
    plt.close()

    # 3. Model Storage Footprint
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    bars = ax.bar(models, sizes, color=colors, width=0.48, edgecolor="#0F172A", linewidth=0.7, zorder=3)
    ax.set_title("Serialized Model Weight Size on Disk", fontsize=11.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_ylabel("Storage Footprint (MB)", fontsize=9.5, fontweight="semibold", color="#334155")
    ax.set_ylim(0, max(sizes) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.7, f"{h:.1f} MB", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax.tick_params(axis="x", labelrotation=12, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "size_comparison.png"))
    plt.close()

    # 4. Accuracy vs Latency Tradeoff
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)
    for i, m in enumerate(models):
        ax.scatter(latencies[i], accuracies[i], s=sizes[i] * 7 + 120, color=colors[i], edgecolors="#0F172A", alpha=0.9, zorder=4)
        offset = (-15, 10) if i == 0 else (10, 6) if i == 1 else (-45, -16) if i == 2 else (10, -12) if i == 3 else (10, 6)
        ax.annotate(
            f"{m}\n({sizes[i]:.1f} MB)",
            (latencies[i], accuracies[i]),
            textcoords="offset points",
            xytext=offset,
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A"
        )
    ax.set_title("Accuracy vs Latency Tradeoff (Area Scaled to Model Size)", fontsize=11.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Inference Latency (ms / sample)", fontsize=9.5, fontweight="semibold", color="#334155")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=9.5, fontweight="semibold", color="#334155")
    ax.set_xlim(2.5, 11)
    ax.set_ylim(83.0, 84.8)
    ax.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tradeoff_pareto.png"))
    plt.close()

    print(f"[✓] Color plots successfully updated in: {output_dir}")

if __name__ == "__main__":
    generate_report_figures()
