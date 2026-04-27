"""
Generate Publication-Ready Figures for Paper
Tổng hợp feature map comparisons thành figures chất lượng cao cho hội nghị khoa học.
"""

import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ===========================================
# CONFIGURATION
# ===========================================

INPUT_DIR = "/home/lqc/Research/Detection/ultralytics/feature_maps_comparison"
OUTPUT_DIR = "/home/lqc/Research/Detection/ultralytics/paper_figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define which visualizations to include
MAIN_COMPARISONS = [
    ("01_FPN_Upsample_mean.jpg", "FPN Upsample"),
    ("02_FPN_C2f_mean.jpg", "FPN C2f"),
    ("03_FPN_Concat_mean.jpg", "FPN Concat"),
    ("04_Neck_Conv_mean.jpg", "Neck Conv/SCDown"),
    ("05_Neck_C2f_mean.jpg", "Neck C2f"),
]

CBAM_FIGURES = [
    ("06_CBAM_1_mean.jpg", "CBAM Attention #1"),
    ("07_CBAM_2_mean.jpg", "CBAM Attention #2"),
]

print("\n" + "=" * 80)
print(" " * 20 + "GENERATING PUBLICATION-READY FIGURES")
print("=" * 80)

# ===========================================
# FIGURE 1: COMPREHENSIVE LAYER COMPARISON
# ===========================================

print("\n[1/4] Creating Figure 1: Comprehensive Feature Map Comparison...")

fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2)

# Load and display input image
input_img = cv2.imread(os.path.join(INPUT_DIR, "00_input_image.jpg"))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

ax_input = fig.add_subplot(gs[0, :])
ax_input.imshow(input_img)
ax_input.set_title("Input Image", fontsize=16, fontweight="bold", pad=20)
ax_input.axis("off")

# Add main comparisons
for idx, (filename, title) in enumerate(MAIN_COMPARISONS):
    row = (idx // 2) + 1
    col = idx % 2

    img_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

fig.suptitle(
    "Feature Map Visualization: YOLOv8 Base vs Optimized (P2+CBAM+SCDown)", fontsize=18, fontweight="bold", y=0.98
)

output_path = os.path.join(OUTPUT_DIR, "Figure1_Comprehensive_Comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ Saved: Figure1_Comprehensive_Comparison.png")

# ===========================================
# FIGURE 2: CBAM ATTENTION VISUALIZATION
# ===========================================

print("\n[2/4] Creating Figure 2: CBAM Attention Modules...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Input image
ax = axes[0, 0]
ax.imshow(input_img)
ax.set_title("(a) Input Image", fontsize=14, fontweight="bold")
ax.axis("off")

# CBAM visualizations
for idx, (filename, title) in enumerate(CBAM_FIGURES):
    row = (idx + 1) // 2
    col = (idx + 1) % 2

    img_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = axes[row, col]
        ax.imshow(img)
        letter = chr(ord("b") + idx)
        ax.set_title(f"({letter}) {title}", fontsize=14, fontweight="bold")
        ax.axis("off")

# Add text explanation
ax = axes[0, 1]
ax.text(
    0.5,
    0.5,
    "CBAM (Convolutional Block Attention Module)\n\n"
    "Key Features:\n"
    "• Spatial attention: WHERE to focus\n"
    "• Channel attention: WHAT features to emphasize\n"
    "• Adaptive feature refinement\n"
    "• Improved small object detection\n\n"
    "Visualizations show focused attention\n"
    "on object regions (red = high attention)",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="center",
    horizontalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax.axis("off")

fig.suptitle("CBAM Attention Module Visualization (Optimized Model)", fontsize=16, fontweight="bold", y=0.98)

output_path = os.path.join(OUTPUT_DIR, "Figure2_CBAM_Attention.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ Saved: Figure2_CBAM_Attention.png")

# ===========================================
# FIGURE 3: GRID VIEW OF ALL METHODS
# ===========================================

print("\n[3/4] Creating Figure 3: Multi-Method Visualization Grid...")

# Select one key layer to show all methods
selected_layer = "03_FPN_Concat"
methods = ["mean", "max", "grid"]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Input image
ax = axes[0, 0]
ax.imshow(input_img)
ax.set_title("(a) Input Image", fontsize=12, fontweight="bold")
ax.axis("off")

# Hide unused subplots in first row
for i in range(1, 3):
    axes[0, i].axis("off")

# Show different visualization methods
for idx, method in enumerate(methods):
    filename = f"{selected_layer}_{method}.jpg"
    img_path = os.path.join(INPUT_DIR, filename)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = axes[1, idx]
        ax.imshow(img)
        letter = chr(ord("b") + idx)
        ax.set_title(f"({letter}) Visualization Method: {method.upper()}", fontsize=12, fontweight="bold")
        ax.axis("off")

fig.suptitle("Feature Map Visualization Methods - FPN Concat Layer", fontsize=16, fontweight="bold", y=0.98)

output_path = os.path.join(OUTPUT_DIR, "Figure3_Visualization_Methods.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ Saved: Figure3_Visualization_Methods.png")

# ===========================================
# FIGURE 4: STATISTICS TABLE
# ===========================================

print("\n[4/4] Creating Figure 4: Quantitative Statistics Table...")

# Statistics from the comparison
statistics = [
    ("FPN_Upsample", 0.028863, 0.160963, 457.68, 0.851426, 0.491514, -42.27),
    ("FPN_C2f", 0.068089, 0.197638, 190.26, 0.586968, 0.493774, -15.88),
    ("FPN_Concat", 0.100399, 0.265800, 164.74, 0.548132, 0.503903, -8.07),
    ("Neck_Conv", 0.201400, 0.333804, 65.74, 0.613027, 0.506475, -17.38),
    ("Neck_C2f", 0.118063, 0.248208, 110.23, 0.640298, 0.524778, -18.04),
]

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("tight")
ax.axis("off")

# Create table
headers = [
    "Layer",
    "Base\nVariance",
    "Opt.\nVariance",
    "Variance\nImprovement",
    "Base\nSparsity",
    "Opt.\nSparsity",
    "Sparsity\nChange",
]

table_data = []
for stat in statistics:
    layer, base_var, opt_var, var_imp, base_sp, opt_sp, sp_change = stat
    table_data.append(
        [
            layer,
            f"{base_var:.4f}",
            f"{opt_var:.4f}",
            f"{var_imp:+.2f}%",
            f"{base_sp:.4f}",
            f"{opt_sp:.4f}",
            f"{sp_change:+.2f}%",
        ]
    )

# Add summary row
avg_var_imp = np.mean([s[3] for s in statistics])
avg_sp_change = np.mean([s[6] for s in statistics])
table_data.append(["AVERAGE", "-", "-", f"{avg_var_imp:+.2f}%", "-", "-", f"{avg_sp_change:+.2f}%"])

table = ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc="center",
    loc="center",
    colWidths=[0.15, 0.12, 0.12, 0.15, 0.12, 0.12, 0.14],
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor("#4472C4")
    cell.set_text_props(weight="bold", color="white")

# Style the data rows
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        if i == len(table_data):  # Summary row
            cell.set_facecolor("#FFC000")
            cell.set_text_props(weight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#D9E2F3")

        # Highlight improvements
        if j == 3:  # Variance improvement column
            val = float(table_data[i - 1][j].strip("%+"))
            if val > 100:
                cell.set_text_props(color="green", weight="bold")
        elif j == 6:  # Sparsity change column
            val = float(table_data[i - 1][j].strip("%+"))
            if val < 0:  # Negative sparsity is good
                cell.set_text_props(color="green", weight="bold")

# Add title and explanations
title_text = "Quantitative Feature Map Analysis: Base vs Optimized Model"
plt.text(0.5, 0.95, title_text, transform=fig.transFigure, ha="center", fontsize=16, fontweight="bold")

explanation = (
    "Channel Variance: Higher values indicate more diverse features (↑ Better)\n"
    "Sparsity: Lower values indicate denser, more informative activations (↓ Better)\n"
    "Key Finding: Optimized model shows 197.73% higher variance and 20.33% lower sparsity on average"
)
plt.text(
    0.5,
    0.08,
    explanation,
    transform=fig.transFigure,
    ha="center",
    fontsize=10,
    style="italic",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
)

output_path = os.path.join(OUTPUT_DIR, "Figure4_Statistics_Table.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ Saved: Figure4_Statistics_Table.png")

# ===========================================
# GENERATE LATEX TABLE
# ===========================================

print("\n[Bonus] Generating LaTeX table...")

latex_output = os.path.join(OUTPUT_DIR, "statistics_table.tex")

with open(latex_output, "w") as f:
    f.write("% LaTeX table for paper\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Quantitative Feature Map Analysis: Base vs Optimized Model}\n")
    f.write("\\label{tab:feature_statistics}\n")
    f.write("\\begin{tabular}{lccccccc}\n")
    f.write("\\hline\n")
    f.write(
        "\\textbf{Layer} & \\multicolumn{3}{c}{\\textbf{Channel Variance}} & \\multicolumn{3}{c}{\\textbf{Sparsity}} \\\\\n"
    )
    f.write("\\cline{2-7}\n")
    f.write(" & Base & Opt. & Improvement & Base & Opt. & Change \\\\\n")
    f.write("\\hline\n")

    for stat in statistics:
        layer, base_var, opt_var, var_imp, base_sp, opt_sp, sp_change = stat
        f.write(
            f"{layer.replace('_', ' ')} & {base_var:.4f} & {opt_var:.4f} & "
            f"+{var_imp:.2f}\\% & {base_sp:.4f} & {opt_sp:.4f} & {sp_change:+.2f}\\% \\\\\n"
        )

    f.write("\\hline\n")
    f.write(
        f"\\textbf{{Average}} & - & - & \\textbf{{+{avg_var_imp:.2f}\\%}} & "
        f"- & - & \\textbf{{{avg_sp_change:+.2f}\\%}} \\\\\n"
    )
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print("✅ Saved: statistics_table.tex")

# ===========================================
# SUMMARY
# ===========================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n📁 Output directory: {OUTPUT_DIR}")
print("\n📊 Generated figures:")
print("   1. Figure1_Comprehensive_Comparison.png - Main comparison of all layers")
print("   2. Figure2_CBAM_Attention.png - CBAM attention visualization")
print("   3. Figure3_Visualization_Methods.png - Different visualization methods")
print("   4. Figure4_Statistics_Table.png - Quantitative analysis table")
print("   5. statistics_table.tex - LaTeX table for paper")

print("\n💡 USAGE IN PAPER:")
print("   • Figure 1: Main results section")
print("   • Figure 2: CBAM module explanation")
print("   • Figure 3: Methodology/supplementary")
print("   • Figure 4: Quantitative results")
print("   • LaTeX table: Copy into your paper source")

print("\n📝 CAPTION TEMPLATE:")
print("""
Figure 1: Feature map visualization comparison between baseline YOLOv8 and 
optimized model (P2+CBAM+SCDown). Left column shows baseline features, right 
column shows optimized model features. The optimized model demonstrates richer 
and more diverse feature representations across all layers (indicated by 
higher variance and denser activations). Visualizations use mean aggregation 
across channels with jet colormap (red=high activation, blue=low activation).
""")

print("\n" + "=" * 80 + "\n")
