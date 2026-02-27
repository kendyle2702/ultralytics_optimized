#!/usr/bin/env python3
"""
Generate LaTeX tables for final evaluation on test set
Reads from 4 JSON result files and creates publication-ready tables.
"""

import json

# Input files
COCO_BASELINE = "no_train_nosahi.json"
COCO_SAHI = "v8_no_train_512_01_test.json"
VISDRONE_BASELINE = "train_nosahi.json"
VISDRONE_SAHI = "512_01_test.json"


def load_json(filepath):
    """Load JSON file."""
    with open(filepath) as f:
        return json.load(f)


def calculate_improvement(new_val, old_val):
    """Calculate percentage improvement."""
    if old_val == 0:
        return 0
    return ((new_val - old_val) / old_val) * 100


def generate_table_with_improvements():
    """Generate table with improvement rows."""
    # Load data
    coco_base = load_json(COCO_BASELINE)
    coco_sahi = load_json(COCO_SAHI)
    vd_base = load_json(VISDRONE_BASELINE)
    vd_sahi = load_json(VISDRONE_SAHI)

    # Extract metrics
    cb_m = coco_base["metrics"]
    cs_m = coco_sahi["metrics"]
    vb_m = vd_base["metrics"]
    vs_m = vd_sahi["metrics"]

    # Calculate improvements
    coco_imp = {
        "mAP50": calculate_improvement(cs_m["mAP50"], cb_m["mAP50"]),
        "mAP50-95": calculate_improvement(cs_m["mAP50-95"], cb_m["mAP50-95"]),
        "mAPs": calculate_improvement(cs_m["mAPs"], cb_m["mAPs"]),
        "mAPm": calculate_improvement(cs_m["mAPm"], cb_m["mAPm"]),
        "mAPl": calculate_improvement(cs_m["mAPl"], cb_m["mAPl"]),
        "FPS": calculate_improvement(cs_m["fps"], cb_m["fps"]),
    }

    vd_imp = {
        "mAP50": calculate_improvement(vs_m["mAP50"], vb_m["mAP50"]),
        "mAP50-95": calculate_improvement(vs_m["mAP50-95"], vb_m["mAP50-95"]),
        "mAPs": calculate_improvement(vs_m["mAPs"], vb_m["mAPs"]),
        "mAPm": calculate_improvement(vs_m["mAPm"], vb_m["mAPm"]),
        "mAPl": calculate_improvement(vs_m["mAPl"], vb_m["mAPl"]),
        "FPS": calculate_improvement(vs_m["fps"], vb_m["fps"]),
    }

    # Generate LaTeX
    latex = r"""\begin{table}[ht]
    \centering
    \caption{Final evaluation on VisDrone test set (1,610 images). SAHI uses optimal configuration (slice=512, overlap=0.1) determined from validation set analysis. All models evaluated with confidence threshold 0.1 and IoU threshold 0.5.}
    \label{tab:final_evaluation}
    \setlength{\tabcolsep}{3.5pt}
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{|l|c|c|c|c|c|c|}
        \hline
        \textbf{Model Configuration} & \textbf{mAP50} & \textbf{mAP50-95} & \textbf{mAPs} & \textbf{mAPm} & \textbf{mAPl} & \textbf{FPS} \\
        \hline
        \hline
        \multicolumn{7}{|l|}{\textit{COCO Pretrained Models (Zero-shot Transfer)}} \\
        \hline
"""

    # COCO rows
    latex += f"        YOLOv8n-COCO & {cb_m['mAP50']:.4f} & {cb_m['mAP50-95']:.4f} & {cb_m['mAPs']:.4f} & {cb_m['mAPm']:.4f} & {cb_m['mAPl']:.4f} & \\textbf{{{cb_m['fps']:.2f}}} \\\\\n"
    latex += f"        \\quad + SAHI (512/0.1) & {cs_m['mAP50']:.4f} & {cs_m['mAP50-95']:.4f} & {cs_m['mAPs']:.4f} & {cs_m['mAPm']:.4f} & {cs_m['mAPl']:.4f} & {cs_m['fps']:.2f} \\\\\n"
    latex += f"        \\quad \\textit{{Improvement}} & \\textit{{{coco_imp['mAP50']:+.0f}\\%}} & \\textit{{{coco_imp['mAP50-95']:+.0f}\\%}} & \\textit{{{coco_imp['mAPs']:+.0f}\\%}} & \\textit{{{coco_imp['mAPm']:+.0f}\\%}} & \\textit{{{coco_imp['mAPl']:+.0f}\\%}} & \\textit{{{coco_imp['FPS']:+.0f}\\%}} \\\\\n"

    latex += r"""        \hline
        \multicolumn{7}{|l|}{\textit{VisDrone Fine-tuned Models}} \\
        \hline
"""

    # VisDrone rows
    latex += f"        YOLOv8n-VisDrone & {vb_m['mAP50']:.4f} & {vb_m['mAP50-95']:.4f} & {vb_m['mAPs']:.4f} & \\textbf{{{vb_m['mAPm']:.4f}}} & \\textbf{{{vb_m['mAPl']:.4f}}} & \\textbf{{{vb_m['fps']:.2f}}} \\\\\n"
    latex += f"        \\quad + SAHI (512/0.1) & \\textbf{{{vs_m['mAP50']:.4f}}} & \\textbf{{{vs_m['mAP50-95']:.4f}}} & \\textbf{{{vs_m['mAPs']:.4f}}} & {vs_m['mAPm']:.4f} & {vs_m['mAPl']:.4f} & {vs_m['fps']:.2f} \\\\\n"
    latex += f"        \\quad \\textit{{Improvement}} & \\textit{{{vd_imp['mAP50']:+.0f}\\%}} & \\textit{{{vd_imp['mAP50-95']:+.0f}\\%}} & \\textit{{{vd_imp['mAPs']:+.0f}\\%}} & \\textit{{{vd_imp['mAPm']:+.0f}\\%}} & \\textit{{{vd_imp['mAPl']:+.0f}\\%}} & \\textit{{{vd_imp['FPS']:+.0f}\\%}} \\\\\n"

    latex += r"""        \hline
    \end{tabular}
\end{table}
"""

    return latex


def generate_table_clean():
    """Generate clean table without improvement rows."""
    # Load data
    coco_base = load_json(COCO_BASELINE)
    coco_sahi = load_json(COCO_SAHI)
    vd_base = load_json(VISDRONE_BASELINE)
    vd_sahi = load_json(VISDRONE_SAHI)

    cb_m = coco_base["metrics"]
    cs_m = coco_sahi["metrics"]
    vb_m = vd_base["metrics"]
    vs_m = vd_sahi["metrics"]

    latex = r"""\begin{table}[ht]
    \centering
    \caption{Final evaluation on VisDrone test set (1,610 images). SAHI configuration (slice=512, overlap=0.1) selected from validation analysis. Bold indicates best performance within each metric category.}
    \label{tab:final_evaluation_clean}
    \setlength{\tabcolsep}{3.5pt}
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{|l|c|c|c|c|c|c|}
        \hline
        \textbf{Model Configuration} & \textbf{mAP50} & \textbf{mAP50-95} & \textbf{mAPs} & \textbf{mAPm} & \textbf{mAPl} & \textbf{FPS} \\
        \hline
        \hline
        \multicolumn{7}{|l|}{\textit{COCO Pretrained Models (Zero-shot Transfer)}} \\
        \hline
"""

    latex += f"        YOLOv8n-COCO & {cb_m['mAP50']:.4f} & {cb_m['mAP50-95']:.4f} & {cb_m['mAPs']:.4f} & {cb_m['mAPm']:.4f} & {cb_m['mAPl']:.4f} & {cb_m['fps']:.2f} \\\\\n"
    latex += f"        \\quad + SAHI (512/0.1) & {cs_m['mAP50']:.4f} & {cs_m['mAP50-95']:.4f} & {cs_m['mAPs']:.4f} & {cs_m['mAPm']:.4f} & {cs_m['mAPl']:.4f} & {cs_m['fps']:.2f} \\\\\n"

    latex += r"""        \hline
        \multicolumn{7}{|l|}{\textit{VisDrone Fine-tuned Models}} \\
        \hline
"""

    latex += f"        YOLOv8n-VisDrone & {vb_m['mAP50']:.4f} & {vb_m['mAP50-95']:.4f} & {vb_m['mAPs']:.4f} & \\textbf{{{vb_m['mAPm']:.4f}}} & \\textbf{{{vb_m['mAPl']:.4f}}} & \\textbf{{{vb_m['fps']:.2f}}} \\\\\n"
    latex += f"        \\quad + SAHI (512/0.1) & \\textbf{{{vs_m['mAP50']:.4f}}} & \\textbf{{{vs_m['mAP50-95']:.4f}}} & \\textbf{{{vs_m['mAPs']:.4f}}} & {vs_m['mAPm']:.4f} & {vs_m['mAPl']:.4f} & {vs_m['fps']:.2f} \\\\\n"

    latex += r"""        \hline
    \end{tabular}
\end{table}
"""

    return latex


def main():
    print(f"\n{'=' * 80}")
    print("📊 Final Table Generator for Test Set Results")
    print(f"{'=' * 80}\n")

    # Generate tables
    print("🔄 Generating LaTeX tables...")

    table1 = generate_table_with_improvements()
    with open("final_table.tex", "w") as f:
        f.write(table1)
    print("✅ Generated: final_table.tex (with improvements)")

    table2 = generate_table_clean()
    with open("final_table_clean.tex", "w") as f:
        f.write(table2)
    print("✅ Generated: final_table_clean.tex (clean version)")

    # Print summary
    print(f"\n{'=' * 80}")
    print("📊 Summary")
    print(f"{'=' * 80}")

    coco_base = load_json(COCO_BASELINE)
    coco_sahi = load_json(COCO_SAHI)
    vd_base = load_json(VISDRONE_BASELINE)
    vd_sahi = load_json(VISDRONE_SAHI)

    print("\nCOCO Models:")
    print(f"  Baseline mAP50: {coco_base['metrics']['mAP50']:.4f}")
    print(
        f"  + SAHI mAP50:   {coco_sahi['metrics']['mAP50']:.4f} ({calculate_improvement(coco_sahi['metrics']['mAP50'], coco_base['metrics']['mAP50']):+.0f}%)"
    )

    print("\nVisDrone Models:")
    print(f"  Baseline mAP50: {vd_base['metrics']['mAP50']:.4f}")
    print(
        f"  + SAHI mAP50:   {vd_sahi['metrics']['mAP50']:.4f} ({calculate_improvement(vd_sahi['metrics']['mAP50'], vd_base['metrics']['mAP50']):+.0f}%)"
    )

    print(f"\n{'=' * 80}")
    print("✅ All tables generated successfully!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
