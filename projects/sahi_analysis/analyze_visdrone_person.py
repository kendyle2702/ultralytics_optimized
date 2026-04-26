#!/usr/bin/env python3
"""
Phân tích kích thước của class Person (Pedestrian + People) trong VisDrone dataset
Vẽ các biểu đồ trình bày phân bố kích thước bounding boxes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Set style for academic publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# VisDrone dataset path
DATASET_ROOT = Path("/home/lqc/Research/Detection/datasets/VisDrone")

# VisDrone class mapping (1-based indexing in file)
CLASS_MAPPING = {
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor'
}

def parse_visdrone_annotations(annotation_file):
    """Parse VisDrone annotation file and extract bounding boxes."""
    bboxes = []
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            score = int(parts[4])
            object_category = int(parts[5])
            truncation = int(parts[6])
            occlusion = int(parts[7])
            
            # Skip ignored regions (score=0) and invalid objects
            if score == 0 or object_category == 0 or object_category > 10:
                continue
            
            bboxes.append({
                'width': bbox_width,
                'height': bbox_height,
                'class': object_category,
                'class_name': CLASS_MAPPING.get(object_category, 'unknown'),
                'truncation': truncation,
                'occlusion': occlusion
            })
    
    return bboxes


def collect_dataset_statistics(splits=['train', 'val', 'test']):
    """Collect statistics from all splits."""
    all_bboxes = []
    
    for split in splits:
        annotation_dir = DATASET_ROOT / f"VisDrone2019-DET-{split}" / "annotations"
        
        if not annotation_dir.exists():
            print(f"⚠️  Warning: {annotation_dir} not found, skipping...")
            continue
        
        print(f"\n📂 Processing {split} split...")
        annotation_files = list(annotation_dir.glob("*.txt"))
        
        for ann_file in tqdm(annotation_files, desc=f"Reading {split}"):
            bboxes = parse_visdrone_annotations(ann_file)
            for bbox in bboxes:
                bbox['split'] = split
            all_bboxes.extend(bboxes)
    
    return all_bboxes


def analyze_person_class(all_bboxes):
    """Analyze Person class (pedestrian + people merged)."""
    
    # Filter for pedestrian (1) and people (2)
    person_bboxes = [bbox for bbox in all_bboxes if bbox['class'] in [1, 2]]
    
    print(f"\n{'='*80}")
    print(f"📊 PHÂN TÍCH CLASS PERSON (Pedestrian + People)")
    print(f"{'='*80}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(person_bboxes)
    
    print(f"\n📈 Tổng quan:")
    print(f"  - Tổng số objects: {len(person_bboxes):,}")
    print(f"  - Pedestrian: {len([b for b in person_bboxes if b['class'] == 1]):,}")
    print(f"  - People: {len([b for b in person_bboxes if b['class'] == 2]):,}")
    
    # Statistics by split
    print(f"\n📊 Phân bố theo split:")
    for split in df['split'].unique():
        count = len(df[df['split'] == split])
        print(f"  - {split.capitalize()}: {count:,} objects")
    
    # Size statistics
    widths = df['width'].values
    heights = df['height'].values
    areas = widths * heights
    aspect_ratios = heights / widths  # height/width ratio
    
    print(f"\n📏 Thống kê kích thước (pixels):")
    print(f"\n  Width (Chiều rộng):")
    print(f"    - Mean: {np.mean(widths):.2f} px")
    print(f"    - Median: {np.median(widths):.2f} px")
    print(f"    - Std: {np.std(widths):.2f} px")
    print(f"    - Min: {np.min(widths):.2f} px")
    print(f"    - Max: {np.max(widths):.2f} px")
    print(f"    - 25th percentile: {np.percentile(widths, 25):.2f} px")
    print(f"    - 75th percentile: {np.percentile(widths, 75):.2f} px")
    
    print(f"\n  Height (Chiều cao):")
    print(f"    - Mean: {np.mean(heights):.2f} px")
    print(f"    - Median: {np.median(heights):.2f} px")
    print(f"    - Std: {np.std(heights):.2f} px")
    print(f"    - Min: {np.min(heights):.2f} px")
    print(f"    - Max: {np.max(heights):.2f} px")
    print(f"    - 25th percentile: {np.percentile(heights, 25):.2f} px")
    print(f"    - 75th percentile: {np.percentile(heights, 75):.2f} px")
    
    print(f"\n  Area (Diện tích):")
    print(f"    - Mean: {np.mean(areas):.2f} px²")
    print(f"    - Median: {np.median(areas):.2f} px²")
    print(f"    - Std: {np.std(areas):.2f} px²")
    
    print(f"\n  Aspect Ratio (Height/Width):")
    print(f"    - Mean: {np.mean(aspect_ratios):.2f}")
    print(f"    - Median: {np.median(aspect_ratios):.2f}")
    print(f"    - Std: {np.std(aspect_ratios):.2f}")
    
    # Size categories
    small_objects = np.sum(areas < 32*32)
    medium_objects = np.sum((areas >= 32*32) & (areas < 96*96))
    large_objects = np.sum(areas >= 96*96)
    
    print(f"\n📦 Phân loại theo kích thước (COCO standard):")
    print(f"  - Small (< 32×32): {small_objects:,} ({small_objects/len(person_bboxes)*100:.1f}%)")
    print(f"  - Medium (32×32 to 96×96): {medium_objects:,} ({medium_objects/len(person_bboxes)*100:.1f}%)")
    print(f"  - Large (> 96×96): {large_objects:,} ({large_objects/len(person_bboxes)*100:.1f}%)")
    
    print(f"\n{'='*80}\n")
    
    return df


def plot_visualizations(df, save_dir="visdrone_analysis"):
    """Create multiple visualization types for academic presentation."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    widths = df['width'].values
    heights = df['height'].values
    
    # =====================================================================
    # PLOT 1: Scatter Plot - Width vs Height
    # Best for: Xem phân bố tương quan giữa width và height
    # =====================================================================
    print("📊 Generating Scatter Plot (Width vs Height)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample data if too large (for better visualization)
    if len(widths) > 10000:
        indices = np.random.choice(len(widths), 10000, replace=False)
        sample_widths = widths[indices]
        sample_heights = heights[indices]
    else:
        sample_widths = widths
        sample_heights = heights
    
    # Create scatter plot with density coloring
    from matplotlib.colors import LinearSegmentedColormap
    hist, xedges, yedges = np.histogram2d(sample_widths, sample_heights, bins=50)
    
    scatter = ax.scatter(sample_widths, sample_heights, 
                        c='#3498db', alpha=0.3, s=10, edgecolors='none')
    
    # Add mean lines
    ax.axvline(np.mean(widths), color='red', linestyle='--', 
               linewidth=2, label=f'Mean Width: {np.mean(widths):.1f}px')
    ax.axhline(np.mean(heights), color='green', linestyle='--', 
               linewidth=2, label=f'Mean Height: {np.mean(heights):.1f}px')
    
    ax.set_xlabel('Width (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('VisDrone Person Class: Bounding Box Dimensions Distribution\n(Pedestrian + People merged)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path / '1_scatter_width_vs_height.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path / '1_scatter_width_vs_height.png'}")
    plt.close()
    
    # =====================================================================
    # PLOT 2: Hexbin Plot - Density Heatmap
    # Best for: Xem phân bố dày đặc với heatmap
    # =====================================================================
    print("📊 Generating Hexbin Density Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    hexbin = ax.hexbin(widths, heights, gridsize=50, cmap='YlOrRd', mincnt=1)
    cb = plt.colorbar(hexbin, ax=ax)
    cb.set_label('Count', fontsize=11, fontweight='bold')
    
    ax.axvline(np.mean(widths), color='blue', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean Width: {np.mean(widths):.1f}px')
    ax.axhline(np.mean(heights), color='blue', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean Height: {np.mean(heights):.1f}px')
    
    ax.set_xlabel('Width (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('VisDrone Person Class: Bounding Box Density Heatmap\n(Pedestrian + People merged)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path / '2_hexbin_density.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path / '2_hexbin_density.png'}")
    plt.close()
    
    # =====================================================================
    # PLOT 3: Histograms - Width and Height Distributions
    # Best for: So sánh phân bố của width và height riêng biệt
    # =====================================================================
    print("📊 Generating Histogram Distributions...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Width distribution
    axes[0].hist(widths, bins=100, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(widths), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(widths):.1f}px')
    axes[0].axvline(np.median(widths), color='green', linestyle='--', 
                    linewidth=2, label=f'Median: {np.median(widths):.1f}px')
    axes[0].set_xlabel('Width (pixels)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Width Distribution of Person Bounding Boxes', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Height distribution
    axes[1].hist(heights, bins=100, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(heights), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(heights):.1f}px')
    axes[1].axvline(np.median(heights), color='green', linestyle='--', 
                    linewidth=2, label=f'Median: {np.median(heights):.1f}px')
    axes[1].set_xlabel('Height (pixels)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Height Distribution of Person Bounding Boxes', 
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path / '3_histogram_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path / '3_histogram_distributions.png'}")
    plt.close()
    
    # =====================================================================
    # PLOT 4: Box Plot - Statistical Summary
    # Best for: So sánh thống kê giữa width và height
    # =====================================================================
    print("📊 Generating Box Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data_to_plot = [widths, heights]
    bp = ax.boxplot(data_to_plot, labels=['Width', 'Height'], patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='green', linewidth=2, linestyle='--'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    ax.set_ylabel('Size (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('VisDrone Person Class: Statistical Summary of Dimensions\n(Pedestrian + People merged)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path / '4_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path / '4_boxplot_comparison.png'}")
    plt.close()
    
    # =====================================================================
    # PLOT 5: Violin Plot - Distribution Shape
    # Best for: Xem shape của distribution (density + box plot)
    # =====================================================================
    print("📊 Generating Violin Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    df_plot = pd.DataFrame({
        'Dimension': ['Width'] * len(widths) + ['Height'] * len(heights),
        'Size': np.concatenate([widths, heights])
    })
    
    violin = ax.violinplot([widths, heights], positions=[1, 2], 
                           showmeans=True, showmedians=True)
    
    # Customize colors
    for pc in violin['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Width', 'Height'])
    ax.set_ylabel('Size (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('VisDrone Person Class: Distribution Shape of Dimensions\n(Pedestrian + People merged)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / '5_violin_plot.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path / '5_violin_plot.png'}")
    plt.close()
    
    # =====================================================================
    # PLOT 6: Combined Analysis - Multi-panel Figure (BEST FOR PAPERS)
    # Best for: Tổng hợp toàn diện cho paper
    # =====================================================================
    print("📊 Generating Combined Multi-panel Figure...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Scatter plot
    ax1 = fig.add_subplot(gs[0, :])
    if len(widths) > 10000:
        indices = np.random.choice(len(widths), 10000, replace=False)
        sample_widths = widths[indices]
        sample_heights = heights[indices]
    else:
        sample_widths = widths
        sample_heights = heights
    ax1.scatter(sample_widths, sample_heights, c='#3498db', alpha=0.3, s=10)
    ax1.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, 
                label=f'Mean Width: {np.mean(widths):.1f}px')
    ax1.axhline(np.mean(heights), color='green', linestyle='--', linewidth=2, 
                label=f'Mean Height: {np.mean(heights):.1f}px')
    ax1.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Scatter Plot: Width vs Height', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Width histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(widths, bins=80, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(widths):.1f}px')
    ax2.axvline(np.median(widths), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(widths):.1f}px')
    ax2.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Width Distribution', fontsize=12, fontweight='bold', loc='left')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Height histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(heights, bins=80, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(heights):.1f}px')
    ax3.axvline(np.median(heights), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(heights):.1f}px')
    ax3.set_xlabel('Height (pixels)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Height Distribution', fontsize=12, fontweight='bold', loc='left')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Box plot
    ax4 = fig.add_subplot(gs[2, 0])
    bp = ax4.boxplot([widths, heights], labels=['Width', 'Height'], patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='green', linewidth=2, linestyle='--'))
    ax4.set_ylabel('Size (pixels)', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Box Plot Comparison', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel E: Statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_data = [
        ['Metric', 'Width (px)', 'Height (px)'],
        ['Mean', f'{np.mean(widths):.2f}', f'{np.mean(heights):.2f}'],
        ['Median', f'{np.median(widths):.2f}', f'{np.median(heights):.2f}'],
        ['Std Dev', f'{np.std(widths):.2f}', f'{np.std(heights):.2f}'],
        ['Min', f'{np.min(widths):.0f}', f'{np.min(heights):.0f}'],
        ['Max', f'{np.max(widths):.0f}', f'{np.max(heights):.0f}'],
        ['25th %ile', f'{np.percentile(widths, 25):.2f}', f'{np.percentile(heights, 25):.2f}'],
        ['75th %ile', f'{np.percentile(widths, 75):.2f}', f'{np.percentile(heights, 75):.2f}'],
    ]
    
    table = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax5.set_title('(E) Statistical Summary', fontsize=12, fontweight='bold', loc='left', pad=20)
    
    # Overall title
    fig.suptitle('VisDrone Person Class (Pedestrian + People): Comprehensive Dimension Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path / '6_combined_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path / '6_combined_analysis.png'}")
    plt.close()
    
    print(f"\n🎉 All visualizations saved to: {save_path.absolute()}/")


def main():
    """Main analysis pipeline."""
    print("🚀 Starting VisDrone Person Class Analysis...")
    print(f"📂 Dataset path: {DATASET_ROOT}")
    
    # Collect data
    all_bboxes = collect_dataset_statistics(splits=['train', 'val', 'test'])
    
    if not all_bboxes:
        print("❌ No bounding boxes found!")
        return
    
    # Analyze person class
    df_person = analyze_person_class(all_bboxes)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    plot_visualizations(df_person)
    
    print("\n✅ Analysis complete!")
    print("\n" + "="*80)
    print("📊 RECOMMENDATION: Which plot to use for presentation?")
    print("="*80)
    print("""
    🏆 FOR ACADEMIC PAPERS: Use #6 (Combined Multi-panel Figure)
       ✓ Comprehensive view with multiple perspectives
       ✓ Professional multi-panel layout
       ✓ Includes statistical table
       ✓ Perfect for paper figures
    
    📈 FOR PRESENTATIONS/SLIDES:
       - Use #2 (Hexbin Density) - Clear and visually appealing
       - Use #3 (Histograms) - Easy to understand
    
    🔬 FOR DETAILED ANALYSIS:
       - Use #1 (Scatter Plot) - Shows individual data points
       - Use #4 (Box Plot) - Statistical comparison
       - Use #5 (Violin Plot) - Distribution shape
    
    💡 TIP: For most papers, use #6 (combined) as main figure,
            and #2 (hexbin) for supplementary materials.
    """)


if __name__ == "__main__":
    main()

