#!/usr/bin/env python3
"""
Visualize COCO metrics results from model evaluation.
Creates comparison charts and tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsVisualizer:
    """Visualize COCO metrics comparison."""

    def __init__(self, results_dir="results/coco_metrics"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (14, 8)
        plt.rcParams["font.size"] = 10

    def load_latest_results(self):
        """Load the most recent results CSV."""
        csv_files = sorted(self.results_dir.glob("coco_metrics_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No results CSV found. Run evaluation first.")

        latest_csv = csv_files[-1]
        print(f"📂 Loading results from: {latest_csv}")

        df = pd.read_csv(latest_csv)
        return df

    def plot_efficiency_scatter(self, df):
        """Plot efficiency scatter: FPS vs AP with params as size."""
        _fig, ax = plt.subplots(figsize=(12, 8))

        # Create scatter plot
        ax.scatter(
            df["FPS"],
            df["AP@[.5:.95]"],
            s=df["params_M"] * 50,  # Size proportional to params
            alpha=0.6,
            c=range(len(df)),
            cmap="viridis",
        )

        # Add model labels
        for idx, row in df.iterrows():
            ax.annotate(
                row["model"],
                (row["FPS"], row["AP@[.5:.95]"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )

        # Formatting
        ax.set_xlabel("FPS (Frames Per Second)", fontsize=12, fontweight="bold")
        ax.set_ylabel("AP@[.5:.95]", fontsize=12, fontweight="bold")
        ax.set_title("Efficiency Analysis: FPS vs AP (bubble size = parameters)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add legend for bubble sizes
        sizes = [10, 25, 50]  # Example params
        labels = [f"{s}M params" for s in sizes]
        legend_elements = [plt.scatter([], [], s=s * 50, alpha=0.6, c="gray") for s in sizes]
        ax.legend(legend_elements, labels, title="Parameters", loc="lower right", fontsize=9)

        plt.tight_layout()
        save_path = self.figures_dir / "efficiency_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_params_fps_comparison(self, df):
        """Plot parameters and FPS comparison."""
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Parameters comparison
        x = np.arange(len(df))
        ax1.bar(x, df["params_M"], color="steelblue", alpha=0.7)
        ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Parameters (Millions)", fontsize=12, fontweight="bold")
        ax1.set_title("Model Size Comparison", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["model"], rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(df["params_M"]):
            ax1.text(i, v + 0.5, f"{v:.2f}M", ha="center", va="bottom", fontsize=9)

        # FPS comparison
        ax2.bar(x, df["FPS"], color="coral", alpha=0.7)
        ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax2.set_ylabel("FPS (Frames Per Second)", fontsize=12, fontweight="bold")
        ax2.set_title("Inference Speed Comparison", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["model"], rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(df["FPS"]):
            ax2.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        save_path = self.figures_dir / "params_fps_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_ap_comparison(self, df):
        """Plot AP metrics comparison bar chart."""
        _fig, ax = plt.subplots(figsize=(14, 6))

        # Select AP metrics
        ap_cols = ["AP@[.5:.95]", "AP@.5", "AP@.75"]

        # Prepare data
        x = np.arange(len(df))
        width = 0.25

        # Plot bars
        for i, col in enumerate(ap_cols):
            ax.bar(x + i * width, df[col], width, label=col)

        # Formatting
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Precision", fontsize=12, fontweight="bold")
        ax.set_title("Average Precision Comparison Across Models", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = self.figures_dir / "ap_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_size_specific_ap(self, df):
        """Plot size-specific AP comparison."""
        _fig, ax = plt.subplots(figsize=(14, 6))

        # Select size-specific AP metrics
        size_cols = ["AP_small", "AP_medium", "AP_large"]

        # Prepare data
        x = np.arange(len(df))
        width = 0.25

        # Plot bars
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        for i, col in enumerate(size_cols):
            ax.bar(x + i * width, df[col], width, label=col.replace("_", " ").title(), color=colors[i])

        # Formatting
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Precision", fontsize=12, fontweight="bold")
        ax.set_title("AP by Object Size (Small Objects Critical for VisDrone)", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = self.figures_dir / "ap_by_size.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_ar_comparison(self, df):
        """Plot AR metrics comparison."""
        _fig, ax = plt.subplots(figsize=(14, 6))

        # Select AR metrics
        ar_cols = ["AR@1", "AR@10", "AR@100"]

        # Prepare data
        x = np.arange(len(df))
        width = 0.25

        # Plot bars
        for i, col in enumerate(ar_cols):
            ax.bar(x + i * width, df[col], width, label=col)

        # Formatting
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Recall", fontsize=12, fontweight="bold")
        ax.set_title("Average Recall Comparison (Max Detections)", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = self.figures_dir / "ar_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_heatmap(self, df):
        """Plot heatmap of all metrics."""
        _fig, ax = plt.subplots(figsize=(16, 8))

        # Prepare data (exclude model column)
        data = df.drop("model", axis=1).T
        data.columns = df["model"]

        # Plot heatmap
        sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "Score"}, ax=ax, linewidths=0.5)

        ax.set_title("COCO Metrics Heatmap - All Models", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric", fontsize=12, fontweight="bold")

        plt.tight_layout()
        save_path = self.figures_dir / "metrics_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_radar_chart(self, df):
        """Plot radar chart for selected models."""
        # Select key metrics for radar chart
        metrics = ["AP@[.5:.95]", "AP@.5", "AP_small", "AR@100", "AR_small"]

        _fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each model
        for idx, row in df.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=row["model"])
            ax.fill(angles, values, alpha=0.15)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_title("Key Metrics Radar Chart", size=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        save_path = self.figures_dir / "radar_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def create_improvement_table(self, df):
        """Create improvement percentage table vs baseline."""
        if len(df) == 0:
            return

        # Assume first model is baseline
        baseline_name = df.iloc[0]["model"]
        baseline_values = df.iloc[0].drop("model")

        # Calculate improvements
        improvements = []
        for idx, row in df.iterrows():
            if idx == 0:
                continue  # Skip baseline

            improvement = {}
            improvement["model"] = row["model"]

            for metric in baseline_values.index:
                baseline_val = baseline_values[metric]
                current_val = row[metric]

                if baseline_val > 0:
                    pct_change = ((current_val - baseline_val) / baseline_val) * 100
                    improvement[metric] = f"{pct_change:+.2f}%"
                else:
                    improvement[metric] = "N/A"

            improvements.append(improvement)

        if improvements:
            imp_df = pd.DataFrame(improvements)

            # Save to file
            table_path = self.figures_dir / "improvement_vs_baseline.txt"
            with open(table_path, "w") as f:
                f.write("=" * 120 + "\n")
                f.write(f"IMPROVEMENT vs BASELINE ({baseline_name})\n")
                f.write("=" * 120 + "\n\n")
                f.write(imp_df.to_string(index=False))
                f.write("\n\n" + "=" * 120 + "\n")

            print(f"✅ Saved: {table_path}")

            # Print to console
            print("\n" + "=" * 120)
            print(f"📊 IMPROVEMENT vs BASELINE ({baseline_name})")
            print("=" * 120)
            print(imp_df.to_string(index=False))
            print("=" * 120)

    def visualize_all(self):
        """Create all visualizations."""
        print("\n" + "=" * 80)
        print("📊 Generating Visualizations")
        print("=" * 80)

        # Load results
        df = self.load_latest_results()

        # Create plots
        print("\n🎨 Creating plots...")
        self.plot_efficiency_scatter(df)
        self.plot_params_fps_comparison(df)
        self.plot_ap_comparison(df)
        self.plot_size_specific_ap(df)
        self.plot_ar_comparison(df)
        self.plot_heatmap(df)
        self.plot_radar_chart(df)

        # Create improvement table
        print("\n📈 Creating improvement table...")
        self.create_improvement_table(df)

        print("\n" + "=" * 80)
        print("✅ All visualizations complete!")
        print(f"📂 Figures saved to: {self.figures_dir}")
        print("=" * 80)


def main():
    """Main function."""
    visualizer = ResultsVisualizer()
    visualizer.visualize_all()


if __name__ == "__main__":
    main()
