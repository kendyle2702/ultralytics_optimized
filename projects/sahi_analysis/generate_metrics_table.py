#!/usr/bin/env python3
"""
Generate metrics table from grid search results
Outputs: CSV, Excel, and Markdown formats.
"""

import json
from pathlib import Path

import pandas as pd

# Configuration
INPUT_FILE = "grid_search_results/all_results.json"
OUTPUT_DIR = Path("grid_search_results")


def load_results(file_path):
    """Load grid search results from JSON."""
    with open(file_path) as f:
        return json.load(f)


def extract_metrics_table(results):
    """Extract metrics into table format."""
    data = []

    # Process baseline
    baseline = results["baseline"]
    data.append(
        {
            "Config": "Baseline",
            "Slice": "-",
            "Overlap": "-",
            "mAP50": baseline["metrics"]["mAP50"],
            "mAP50-95": baseline["metrics"]["mAP50-95"],
            "mAPs": baseline["metrics"]["mAPs"],
            "mAPm": baseline["metrics"]["mAPm"],
            "mAPl": baseline["metrics"]["mAPl"],
            "FPS": baseline["metrics"]["fps"],
        }
    )

    # Process SAHI configs
    for key, result in results.items():
        if key == "baseline" or result is None:
            continue

        # Extract slice and overlap from config
        config = result["config"]
        slice_size = config["slice_size"]
        overlap = config["overlap_ratio"]

        data.append(
            {
                "Config": f"SAHI {slice_size}/{overlap}",
                "Slice": slice_size,
                "Overlap": overlap,
                "mAP50": result["metrics"]["mAP50"],
                "mAP50-95": result["metrics"]["mAP50-95"],
                "mAPs": result["metrics"]["mAPs"],
                "mAPm": result["metrics"]["mAPm"],
                "mAPl": result["metrics"]["mAPl"],
                "FPS": result["metrics"]["fps"],
            }
        )

    return pd.DataFrame(data)


def save_table(df, output_dir):
    """Save table in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_file = output_dir / "metrics_table.csv"
    df.to_csv(csv_file, index=False, float_format="%.4f")
    print(f"✅ CSV saved: {csv_file}")

    # Excel
    excel_file = output_dir / "metrics_table.xlsx"
    df.to_excel(excel_file, index=False, float_format="%.4f")
    print(f"✅ Excel saved: {excel_file}")

    # Markdown
    md_file = output_dir / "metrics_table.md"
    with open(md_file, "w") as f:
        f.write("# Grid Search Metrics Table\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"✅ Markdown saved: {md_file}")

    # LaTeX
    latex_file = output_dir / "metrics_table.tex"
    with open(latex_file, "w") as f:
        latex_str = df.to_latex(index=False, float_format="%.4f")
        f.write(latex_str)
    print(f"✅ LaTeX saved: {latex_file}")

    return csv_file, excel_file, md_file, latex_file


def print_summary(df):
    """Print summary statistics."""
    print(f"\n{'=' * 80}")
    print("📊 Summary Statistics")
    print(f"{'=' * 80}")
    print(f"Total configs: {len(df)}")
    print("  - Baseline: 1")
    print(f"  - SAHI: {len(df) - 1}")

    # Best mAP50
    best_map50_idx = df["mAP50"].idxmax()
    print("\n🏆 Best mAP50:")
    print(f"  Config: {df.loc[best_map50_idx, 'Config']}")
    print(f"  mAP50: {df.loc[best_map50_idx, 'mAP50']:.4f}")
    print(f"  FPS: {df.loc[best_map50_idx, 'FPS']:.2f}")

    # Best mAPs (small objects)
    best_maps_idx = df[df["Config"] != "Baseline"]["mAPs"].idxmax()
    print("\n⭐ Best Small Object Detection (mAPs):")
    print(f"  Config: {df.loc[best_maps_idx, 'Config']}")
    print(f"  mAPs: {df.loc[best_maps_idx, 'mAPs']:.4f}")
    print(f"  FPS: {df.loc[best_maps_idx, 'FPS']:.2f}")

    # Best FPS (SAHI only)
    sahi_df = df[df["Config"] != "Baseline"]
    best_fps_idx = sahi_df["FPS"].idxmax()
    print("\n🚀 Fastest SAHI Config:")
    print(f"  Config: {df.loc[best_fps_idx, 'Config']}")
    print(f"  FPS: {df.loc[best_fps_idx, 'FPS']:.2f}")
    print(f"  mAP50: {df.loc[best_fps_idx, 'mAP50']:.4f}")

    # Baseline comparison
    baseline_map50 = df.loc[0, "mAP50"]
    baseline_fps = df.loc[0, "FPS"]
    best_improvement = ((df.loc[best_map50_idx, "mAP50"] - baseline_map50) / baseline_map50) * 100
    print("\n📈 Best Improvement vs Baseline:")
    print(f"  mAP50: +{best_improvement:.1f}%")
    print(f"  FPS: {((df.loc[best_map50_idx, 'FPS'] - baseline_fps) / baseline_fps) * 100:.1f}%")

    print(f"{'=' * 80}\n")


def main():
    print(f"\n{'=' * 80}")
    print("📊 Grid Search Metrics Table Generator")
    print(f"{'=' * 80}\n")

    # Load results
    print(f"📂 Loading: {INPUT_FILE}")
    results = load_results(INPUT_FILE)
    print(f"✅ Loaded {len(results)} configurations\n")

    # Extract metrics
    print("🔄 Extracting metrics...")
    df = extract_metrics_table(results)
    print(f"✅ Extracted {len(df)} rows\n")

    # Save in multiple formats
    print("💾 Saving tables...")
    save_table(df, OUTPUT_DIR)

    # Print summary
    print_summary(df)

    # Display first few rows
    print("📋 Preview (first 10 rows):")
    print(df.head(10).to_string(index=False))
    print(f"\n... and {len(df) - 10} more rows\n")

    print(f"{'=' * 80}")
    print("✅ All tables generated successfully!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
