#!/usr/bin/env python3
"""
Visualize Grid Search Results - 3 Panel Figure (A, B, C only)
Loads results from grid_search_results/progress.json or all_results.json
Creates clean publication-ready figure with 3 subplots
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Configuration
RESULTS_DIR = Path("grid_search_results")
OUTPUT_FILE = RESULTS_DIR / "grid_search_3panel.png"

# Try to load results
RESULTS_FILE = RESULTS_DIR / "all_results.json"
if not RESULTS_FILE.exists():
    RESULTS_FILE = RESULTS_DIR / "progress.json"

if not RESULTS_FILE.exists():
    print(f"❌ Error: No results file found in {RESULTS_DIR}")
    print("   Expected: all_results.json or progress.json")
    exit(1)


def load_results():
    """Load grid search results from JSON file."""
    print(f"📂 Loading results from: {RESULTS_FILE}")
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    print(f"✅ Loaded {len(results)} experiment results")
    return results


def extract_metrics_table(results):
    """Extract metrics into structured format for visualization."""
    data = {
        'config': [],
        'slice_size': [],
        'overlap': [],
        'mAP50': [],
        'mAP50-95': [],
        'mAPs': [],
        'mAPm': [],
        'mAPl': [],
        'fps': [],
    }
    
    # Add baseline
    if 'baseline' not in results:
        print("⚠️  Warning: No baseline results found")
        return None
    
    baseline = results['baseline']
    data['config'].append('Baseline')
    data['slice_size'].append(0)
    data['overlap'].append(0)
    data['mAP50'].append(baseline['metrics']['mAP50'])
    data['mAP50-95'].append(baseline['metrics']['mAP50-95'])
    data['mAPs'].append(baseline['metrics']['mAPs'])
    data['mAPm'].append(baseline['metrics']['mAPm'])
    data['mAPl'].append(baseline['metrics']['mAPl'])
    data['fps'].append(baseline['metrics']['fps'])
    
    # Add SAHI results
    sahi_count = 0
    for key, result in results.items():
        if key == 'baseline':
            continue
        
        # Skip failed experiments
        if result is None:
            print(f"⚠️  Skipping failed experiment: {key}")
            continue
        
        # Get config from result (more robust than parsing key)
        try:
            slice_size = result['config']['slice_size']
            overlap = result['config']['overlap_ratio']
            
            data['config'].append(f'SAHI {slice_size}/{overlap}')
            data['slice_size'].append(slice_size)
            data['overlap'].append(overlap)
            data['mAP50'].append(result['metrics']['mAP50'])
            data['mAP50-95'].append(result['metrics']['mAP50-95'])
            data['mAPs'].append(result['metrics']['mAPs'])
            data['mAPm'].append(result['metrics']['mAPm'])
            data['mAPl'].append(result['metrics']['mAPl'])
            data['fps'].append(result['metrics']['fps'])
            
            sahi_count += 1
        except (ValueError, KeyError, IndexError) as e:
            print(f"⚠️  Error parsing {key}: {e}")
            continue
    
    print(f"✅ Extracted {sahi_count} SAHI experiment results")
    return data


def create_3panel_figure(data):
    """Create 3-panel figure with A, B, C subplots."""
    
    # Create figure with 1 row, 3 columns
    fig = plt.figure(figsize=(15, 4.5))
    
    # Prepare SAHI data
    sahi_indices = [i for i, s in enumerate(data['slice_size']) if s > 0]
    slice_sizes_sahi = [data['slice_size'][i] for i in sahi_indices]
    overlaps_sahi = [data['overlap'][i] for i in sahi_indices]
    mAP50_sahi = [data['mAP50'][i] for i in sahi_indices]
    
    unique_slices = sorted(set(slice_sizes_sahi))
    unique_overlaps = sorted(set(overlaps_sahi))
    
    print(f"\n📊 Data summary:")
    print(f"   Slice sizes: {unique_slices}")
    print(f"   Overlap ratios: {unique_overlaps}")
    print(f"   Total SAHI configs: {len(sahi_indices)}")
    
    # ========================================================================
    # Subplot A: Heatmap - mAP50 vs Slice Size vs Overlap
    # ========================================================================
    ax1 = plt.subplot(1, 3, 1)
    
    # Create pivot table for heatmap
    heatmap_data = np.zeros((len(unique_overlaps), len(unique_slices)))
    for i, overlap in enumerate(unique_overlaps):
        for j, slice_size in enumerate(unique_slices):
            # Find matching config
            matches = [m for k, (s, o, m) in enumerate(zip(slice_sizes_sahi, overlaps_sahi, mAP50_sahi)) 
                      if s == slice_size and o == overlap]
            if matches:
                heatmap_data[i, j] = matches[0]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=unique_slices, yticklabels=unique_overlaps,
                cbar_kws={'label': 'mAP50'}, ax=ax1, vmin=0.4, vmax=0.6)
    ax1.set_xlabel('Slice Size (pixels)', fontsize=11, weight='bold')
    ax1.set_ylabel('Overlap Ratio', fontsize=11, weight='bold')
    ax1.set_title('(A) mAP50 Heatmap: Hyperparameter Grid', fontsize=12, weight='bold', pad=10)
    
    # ========================================================================
    # Subplot B: Line Plot - mAP50 by Slice Size for each Overlap
    # ========================================================================
    ax2 = plt.subplot(1, 3, 2)
    
    # Add baseline
    baseline_mAP50 = data['mAP50'][0]
    ax2.axhline(y=baseline_mAP50, color='black', linestyle='--', linewidth=2.5, 
                label='Baseline (No SAHI)', alpha=0.8, zorder=1)
    
    # Plot each overlap ratio
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_overlaps)))
    for i, overlap in enumerate(unique_overlaps):
        # Get data for this overlap
        indices = [k for k, (s, o) in enumerate(zip(slice_sizes_sahi, overlaps_sahi)) if o == overlap]
        slices = [slice_sizes_sahi[k] for k in indices]
        mAPs = [mAP50_sahi[k] for k in indices]
        
        # Sort by slice size
        sorted_data = sorted(zip(slices, mAPs))
        if sorted_data:
            slices, mAPs = zip(*sorted_data)
            ax2.plot(slices, mAPs, marker='o', linewidth=2.5, markersize=8, 
                    label=f'Overlap={overlap:.1f}', color=colors[i], zorder=2)
    
    ax2.set_xlabel('Slice Size (pixels)', fontsize=11, weight='bold')
    ax2.set_ylabel('mAP50', fontsize=11, weight='bold')
    ax2.set_title('(B) Overall Detection Performance', fontsize=12, weight='bold', pad=10)
    ax2.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.4, 0.6])
    
    # ========================================================================
    # Subplot C: Scatter Plot - Speed-Accuracy Tradeoff
    # ========================================================================
    ax3 = plt.subplot(1, 3, 3)
    
    # Baseline - larger and more prominent
    ax3.scatter(data['fps'][0], data['mAP50'][0], s=400, marker='*', 
                c='gold', edgecolors='black', linewidth=2.5, 
                label='Baseline (No SAHI)', zorder=10)
    
    # SAHI configurations - colored by overlap ratio
    for i, overlap in enumerate(unique_overlaps):
        indices = [k for k in sahi_indices if data['overlap'][k] == overlap]
        fps_vals = [data['fps'][k] for k in indices]
        mAP50_vals = [data['mAP50'][k] for k in indices]
        
        ax3.scatter(fps_vals, mAP50_vals, s=150, alpha=0.8, 
                   label=f'Overlap={overlap:.1f}', color=colors[i],
                   edgecolors='black', linewidth=1.0, zorder=5)
    
    ax3.set_xlabel('FPS (frames per second)', fontsize=11, weight='bold')
    ax3.set_ylabel('mAP50', fontsize=11, weight='bold')
    ax3.set_title('(C) Speed-Accuracy Tradeoff', fontsize=12, weight='bold', pad=10)
    ax3.set_xscale('log')
    
    # Customize x-axis ticks - more values and normal format (not scientific)
    from matplotlib.ticker import FuncFormatter
    ax3.set_xticks([3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60])
    ax3.set_xticklabels(['3', '4', '5', '6', '8', '10', '15', '20', '30', '40', '60'])
    ax3.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    
    ax3.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, which='both', linestyle='--')
    ax3.set_ylim([0.4, 0.6])
    
    # Add Pareto frontier annotation
    # ax3.annotate('Slower but\nmore accurate', xy=(0.05, 0.95), xycoords='axes fraction',
    #             fontsize=9, ha='left', va='top', style='italic', 
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    # ax3.annotate('Faster but\nless accurate', xy=(0.95, 0.05), xycoords='axes fraction',
    #             fontsize=9, ha='right', va='bottom', style='italic',
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    # ========================================================================
    # Save figure
    # ========================================================================
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✅ 3-panel figure saved to: {OUTPUT_FILE}")
    plt.close()
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"   Baseline mAP50: {baseline_mAP50:.4f}")
    print(f"   Best SAHI mAP50: {max(mAP50_sahi):.4f} (+{((max(mAP50_sahi)/baseline_mAP50)-1)*100:.1f}%)")
    print(f"   Baseline FPS: {data['fps'][0]:.2f}")
    best_mAP_idx = sahi_indices[np.argmax(mAP50_sahi)]
    print(f"   Best config: slice={data['slice_size'][best_mAP_idx]}, overlap={data['overlap'][best_mAP_idx]}")
    print(f"   Best config FPS: {data['fps'][best_mAP_idx]:.2f} ({((data['fps'][best_mAP_idx]/data['fps'][0])-1)*100:.1f}% vs baseline)")


def main():
    print(f"\n{'='*80}")
    print("📊 Grid Search Results Visualization (3-Panel Figure)")
    print(f"{'='*80}\n")
    
    # Load results
    results = load_results()
    
    # Extract metrics
    print("\n🔄 Extracting metrics...")
    data = extract_metrics_table(results)
    
    if data is None:
        print("❌ Failed to extract metrics")
        return
    
    # Create visualization
    print("\n🎨 Creating 3-panel figure...")
    create_3panel_figure(data)
    
    print(f"\n{'='*80}")
    print("✅ Visualization complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

