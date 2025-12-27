#!/usr/bin/env python3
"""
Tạo biểu đồ compact version cho paper - chỉ 3 panels chính
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set style for academic publication
plt.style.use('seaborn-v0_8-paper')

def create_compact_figure():
    """Create compact 3-panel figure for paper."""
    
    # Load data from previous analysis
    analysis_dir = Path("/home/lqc/Research/Detection/ultralytics/visdrone_analysis")
    
    # Re-read the data or use saved statistics
    # For now, let's reconstruct from the dataset
    from analyze_visdrone_person import collect_dataset_statistics
    
    print("📊 Loading dataset statistics...")
    all_bboxes = collect_dataset_statistics(splits=['train', 'val'])
    
    # Filter for person class
    person_bboxes = [bbox for bbox in all_bboxes if bbox['class'] in [1, 2]]
    df = pd.DataFrame(person_bboxes)
    
    widths = df['width'].values
    heights = df['height'].values
    
    print(f"✅ Loaded {len(person_bboxes):,} person objects")
    
    # =====================================================================
    # Create 3-panel figure (only A, B, C)
    # =====================================================================
    print("🎨 Generating compact 3-panel figure...")
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Scatter plot (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    if len(widths) > 10000:
        indices = np.random.choice(len(widths), 10000, replace=False)
        sample_widths = widths[indices]
        sample_heights = heights[indices]
    else:
        sample_widths = widths
        sample_heights = heights
    
    ax1.scatter(sample_widths, sample_heights, c='#3498db', alpha=0.3, s=10, edgecolors='none')
    ax1.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean Width: {np.mean(widths):.1f}px')
    ax1.axhline(np.mean(heights), color='green', linestyle='--', linewidth=2.5, 
                label=f'Mean Height: {np.mean(heights):.1f}px')
    ax1.set_xlabel('Width (pixels)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Height (pixels)', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Bounding Box Dimensions: Width vs Height', 
                  fontsize=14, fontweight='bold', loc='left', pad=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: Width histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(widths, bins=80, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {np.mean(widths):.1f}px')
    ax2.axvline(np.median(widths), color='green', linestyle='--', linewidth=2.5, 
                label=f'Median: {np.median(widths):.1f}px')
    ax2.set_xlabel('Width (pixels)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Width Distribution', fontsize=14, fontweight='bold', loc='left', pad=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Panel C: Height histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(heights, bins=80, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(np.mean(heights), color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {np.mean(heights):.1f}px')
    ax3.axvline(np.median(heights), color='green', linestyle='--', linewidth=2.5, 
                label=f'Median: {np.median(heights):.1f}px')
    ax3.set_xlabel('Height (pixels)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax3.set_title('(C) Height Distribution', fontsize=14, fontweight='bold', loc='left', pad=10)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle('VisDrone Person Class (Pedestrian + People): Dimension Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = analysis_dir / '6_combined_analysis_compact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved compact figure: {output_path}")
    
    # Also save with different sizes for flexibility
    output_path_medium = analysis_dir / '6_combined_analysis_compact_medium.png'
    plt.savefig(output_path_medium, dpi=200, bbox_inches='tight')
    print(f"✅ Saved medium resolution: {output_path_medium}")
    
    output_path_small = analysis_dir / '6_combined_analysis_compact_small.png'
    plt.savefig(output_path_small, dpi=150, bbox_inches='tight')
    print(f"✅ Saved small resolution: {output_path_small}")
    
    plt.close()
    
    # Print statistics for reference
    print("\n" + "="*70)
    print("📊 STATISTICS SUMMARY (for caption/text)")
    print("="*70)
    print(f"Total objects: {len(person_bboxes):,}")
    print(f"\nWidth:  Mean={np.mean(widths):.2f}px, Median={np.median(widths):.2f}px, Std={np.std(widths):.2f}px")
    print(f"Height: Mean={np.mean(heights):.2f}px, Median={np.median(heights):.2f}px, Std={np.std(heights):.2f}px")
    print(f"Aspect Ratio (H/W): {np.mean(heights/widths):.2f}")
    
    areas = widths * heights
    small_pct = np.sum(areas < 32*32) / len(areas) * 100
    print(f"\nSmall objects (< 32×32px): {small_pct:.1f}%")
    print("="*70)
    
    print("\n💡 TIP: Use 'compact.png' for main paper, 'compact_medium.png' for slides")


if __name__ == "__main__":
    create_compact_figure()

