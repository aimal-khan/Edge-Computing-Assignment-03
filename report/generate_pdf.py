import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)

def build_pdf_report(
    pdf_path: str = "./report/Technical_Report_i211701_A.pdf",
    plots_dir: str = "./report/plots"
):
    """Compile formal technical documentation into a clean academic PDF with bold black headings."""
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Pure bold black and crisp monochrome typographic hierarchy
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#000000"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#000000"),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#000000"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#000000"),
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#111111"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#000000"),
        alignment=1,
        spaceAfter=8
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Edge Model Optimization: Quantization, Pruning & Low-Latency Inference", title_style))
    meta_text = (
        "<b>Project:</b> Edge-Optimized Neural Compression Suite &nbsp;|&nbsp; "
        "<b>Affiliation:</b> FAST NUCES &nbsp;|&nbsp; "
        "<b>Author:</b> Aimal Khan (Roll: <b>i211701</b>, Sec: <b>A</b>)<br/>"
        "<b>Evaluation Platform:</b> Apple M1 (ARM64, 16 GB Unified Memory) &nbsp;|&nbsp; "
        "<b>GitHub:</b> https://github.com/aimal-khan/Edge-Computing-Assignment-03"
    )
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#000000"), spaceAfter=10))

    # 1. Hardware Context & Motivation
    story.append(Paragraph("1. Hardware Context & Roofline Bottlenecks", h1_style))
    p1 = (
        "On embedded and mobile microprocessors, running deep neural networks is fundamentally memory-bound rather than compute-bound. "
        "Fetching full-precision 32-bit floating-point weights across a narrow memory bus from DRAM burns substantially more energy than "
        "the multiply-accumulate operations themselves. A standard ResNet-18 contains 11.17 million parameters, totaling 42.7 MB in FP32. "
        "When processing continuous camera streams or real-time sensor frames on edge nodes, this large footprint repeatedly thrashes L1 and L2 caches, "
        "leading to thermal throttling and high per-frame latency. This project implements and systematically compares four distinct compression "
        "methods on CIFAR-10: Post-Training Static Quantization (PTQ), Quantization-Aware Training (QAT), iterative unstructured pruning, and "
        "structured channel pruning with full tensor reconstruction. All models were tested on the complete 10,000-image evaluation set."
    )
    story.append(Paragraph(p1, body_style))

    # 2. Consolidated Benchmark Summary Table
    story.append(Paragraph("2. Consolidated Empirical Benchmark Results", h1_style))
    table_data = [
        ["Model Variant", "Top-1 Acc", "CPU Latency", "File Size", "Parameters", "Sparsity", "Param Delta"],
        ["FP32 Baseline", "83.62%", "9.30 ms", "42.70 MB", "11,173,962", "0.00%", "Baseline"],
        ["PTQ-INT8", "83.62%", "3.75 ms", "10.71 MB", "11,173,962", "0.00%", "3.99x Storage"],
        ["QAT-INT8", "83.99%", "3.67 ms", "10.71 MB", "11,173,962", "0.00%", "3.99x Storage"],
        ["Unstructured Pruned", "84.35%", "9.25 ms", "42.70 MB", "11,173,962", "49.96%", "0.00% Dense"],
        ["Structured Pruned", "83.52%", "7.45 ms", "30.09 MB", "7,864,834", "0.00%", "-29.61% Physical"]
    ]
    t = Table(table_data, colWidths=[115, 62, 65, 65, 75, 58, 68])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#000000")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f8f8"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#333333")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # 3. Baseline Adaptation to CIFAR-10
    story.append(Paragraph("3. Baseline ResNet-18 Adaptation to CIFAR-10", h1_style))
    p_base = (
        "Standard torchvision ResNet-18 uses a 7x7 conv with stride 2 and max-pooling designed for 224x224 ImageNet inputs. "
        "Applying that stem to 32x32 CIFAR-10 inputs immediately collapses spatial resolution down to 8x8 in block 1. "
        "We substituted this with a 3x3 convolution with stride 1 and padding 1, replaced max-pooling with nn.Identity(), "
        "and replaced the 1000-class head with a 10-class linear layer. Training for 5 epochs with SGD (momentum 0.9, cosine schedule) "
        "yielded 83.62% test accuracy, providing a solid baseline checkpoint for both quantization and pruning pipelines."
    )
    story.append(Paragraph(p_base, body_style))

    # 4. Integer Quantization Analysis
    story.append(Paragraph("4. Integer Quantization Pipeline: PTQ vs. QAT", h1_style))
    story.append(Paragraph("4.1 Post-Training Static Quantization (PTQ)", h2_style))
    p_ptq = (
        "PTQ maps continuous floating-point tensors into 8-bit integers via affine scaling: q = clamp(round(x/S) + Z, -128, 127). "
        "Before inserting observers, our implementation fuses Conv2d + BatchNorm2d + ReLU blocks, absorbing batch normalization parameters "
        "directly into the convolutional kernels. On our Apple Silicon M1 testbed, PyTorch requires setting the backend engine to 'qnnpack', "
        "as x86 'fbgemm' fails immediately with NoQEngine. After feeding 500 unaugmented training calibration images to establish activation "
        "histograms, integer conversion reduced model storage from 42.70 MB down to 10.71 MB (a 3.99x reduction) and cut CPU latency from "
        "9.30 ms down to 3.75 ms (a 2.48x speedup) with zero accuracy degradation (83.62%)."
    )
    story.append(Paragraph(p_ptq, body_style))

    story.append(Paragraph("4.2 Quantization-Aware Training (QAT)", h2_style))
    p_qat = (
        "QAT models quantization noise directly during backpropagation. Fake-quantization operators simulate integer rounding and clamping "
        "during the forward pass, while gradients bypass discretization via the Straight-Through Estimator (STE). "
        "Fine-tuning for 50 batches with SGD (lr 1e-4) allowed the weights to adapt to rounding boundaries. Freezing batch normalization "
        "statistics prior to conversion produced an INT8 checkpoint with 83.99% accuracy (+0.37% over baseline) and a per-image CPU latency "
        "of 3.67 ms (a 2.53x speedup), outperforming both the original unquantized model and static PTQ."
    )
    story.append(Paragraph(p_qat, body_style))

    # 5. Pruning Analysis
    story.append(Paragraph("5. Neural Network Pruning Framework", h1_style))
    story.append(Paragraph("5.1 Iterative Unstructured Magnitude Pruning", h2_style))
    p_unstr = (
        "Unstructured pruning zeros individual weights based on absolute L1-norm magnitude. To reach 50% target sparsity without severe "
        "accuracy collapse, we implemented an iterative 3-round schedule (20.6% -> 37.0% -> 50.0%) with intermediate fine-tuning. "
        "Calling prune.remove() made the zero-masks permanent, achieving 49.96% sparsity and 84.35% accuracy. However, single-image latency "
        "remained unchanged (9.25 ms vs. 9.30 ms baseline) because standard dense CPU BLAS routines execute zero-multiplications at full cost "
        "without specialized sparse tensor hardware."
    )
    story.append(Paragraph(p_unstr, body_style))

    story.append(Paragraph("5.2 Structured Channel Pruning & Physical Reconstruction", h2_style))
    p_str = (
        "Structured channel pruning removes entire 2D feature maps to force execution speedups on standard CPUs. We evaluated filter importance "
        "using L2-norms and pruned the bottom 30% of channels in each residual block's conv1 layer. Because conv1's output channels feed conv2's "
        "input, we sliced conv2's input weights accordingly while preserving conv2's output dimensions, ensuring the residual addition x + F(x) "
        "remained an exact dimensional match. Crucially, we physically reconstructed new, smaller nn.Conv2d and nn.BatchNorm2d layers rather than "
        "relying on zero masks. This permanently cut parameters from 11.17M to 7.86M (-29.61%), shrank disk size to 30.09 MB, restored accuracy "
        "to 83.52% after fine-tuning, and achieved a genuine 20% latency drop on CPU (7.45 ms vs. 9.30 ms)."
    )
    story.append(Paragraph(p_str, body_style))

    # 6. Visual Performance Summaries
    story.append(Paragraph("6. Empirical Visual Summaries", h1_style))
    
    img_width = 240
    img_height = 135

    p_acc = os.path.join(plots_dir, "accuracy_comparison.png")
    p_lat = os.path.join(plots_dir, "latency_comparison.png")
    p_size = os.path.join(plots_dir, "size_comparison.png")
    p_pareto = os.path.join(plots_dir, "tradeoff_pareto.png")

    if os.path.exists(p_acc) and os.path.exists(p_lat):
        row1 = [
            [Image(p_acc, width=img_width, height=img_height), Image(p_lat, width=img_width, height=img_height)],
            [Paragraph("Figure 1: Test Accuracy Across Paradigms", caption_style),
             Paragraph("Figure 2: CPU Latency per Image (Apple M1)", caption_style)]
        ]
        t_row1 = Table(row1, colWidths=[250, 250])
        t_row1.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        story.append(t_row1)
        story.append(Spacer(1, 4))

    if os.path.exists(p_size) and os.path.exists(p_pareto):
        row2 = [
            [Image(p_size, width=img_width, height=img_height), Image(p_pareto, width=img_width, height=img_height)],
            [Paragraph("Figure 3: Serialized Disk Storage (MB)", caption_style),
             Paragraph("Figure 4: Accuracy vs Latency Pareto Frontier", caption_style)]
        ]
        t_row2 = Table(row2, colWidths=[250, 250])
        t_row2.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        story.append(t_row2)
        story.append(Spacer(1, 8))

    # 7. Deployment Guidelines
    story.append(Paragraph("7. Practical Engineering Takeaways for Edge Deployment", h1_style))
    recs = [
        "<b>Quantization is the Most Effective Single Step:</b> Lowering precision to INT8 yields an instant 4x memory reduction and 2.5x speedup with zero accuracy loss on ARM CPUs.",
        "<b>QAT Outperforms Static Calibration:</b> Simulating quantization in training allows weights to adapt to discretization noise, lifting accuracy (+0.37% over baseline).",
        "<b>Unstructured Pruning Does Not Accelerate Standard CPUs:</b> Zero-weight masking leaves tensor shapes unchanged; dense CPU BLAS routines still compute every zero.",
        "<b>Structured Channel Reconstruction Yields Real Hardware Speedups:</b> Physically reducing tensor dimensions eliminated 3.3 million parameters and cut latency by 20%.",
        "<b>Recommended Compound Pipeline:</b> First apply structured channel pruning to physically slim the graph, then apply QAT on the reconstructed architecture for maximum speed and density."
    ]
    for r in recs:
        story.append(Paragraph(f"• {r}", bullet_style))

    doc.build(story)
    print(f"[✓] Bold black heading PDF successfully compiled: {pdf_path}")

if __name__ == "__main__":
    build_pdf_report()
