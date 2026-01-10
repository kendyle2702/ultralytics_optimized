#!/usr/bin/env python3
"""
Grid Search for SAHI Hyperparameters on VisDrone
Scientifically evaluates slice_size and overlap_ratio combinations
Generates publication-ready visualizations
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# ============================================================================
# GRID SEARCH CONFIGURATION (Fixed Parameters)
# ============================================================================

# Model and dataset configuration
MODEL_PATH = "v8_trained.pt"  # Change this to your trained model
DATASET_ROOT = "/home/lqc/Research/Detection/datasets/VisDrone"
SPLIT = "val"  # Use validation set for hyperparameter tuning

# Fixed inference parameters
CONF_THRESHOLD = 0.4  # Low threshold to capture all detections
IOU_THRESHOLD = 0.5   # Standard NMS threshold
IMGSZ = 640           # Standard YOLO input size
DEVICE = "cuda"

# Grid search parameters
SLICE_SIZES = [256, 320, 384, 448, 512, 640, 768]
OVERLAP_RATIOS = [0.1, 0.2, 0.3, 0.4]

# Output directory
OUTPUT_DIR = Path("grid_search_results_conf_0.4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_baseline() -> Dict:
    """Run baseline evaluation without SAHI."""
    print(f"\n{'='*80}")
    print("🔬 Running BASELINE (No SAHI)")
    print(f"{'='*80}\n")
    
    output_file = OUTPUT_DIR / "baseline.json"
    
    if output_file.exists():
        print(f"✅ Baseline results already exist, loading from {output_file}")
        with open(output_file, 'r') as f:
            return json.load(f)
    
    cmd = [
        "python", "evaluate_visdrone_pipeline.py",
        "--model", MODEL_PATH,
        "--dataset", DATASET_ROOT,
        "--split", SPLIT,
        "--no-sahi",
        "--imgsz", str(IMGSZ),
        "--conf", str(CONF_THRESHOLD),
        "--iou", str(IOU_THRESHOLD),
        "--device", DEVICE,
        "--output", str(output_file)
    ]
    
    subprocess.run(cmd, check=True)
    
    # Clear GPU memory after baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    time.sleep(2)
    
    with open(output_file, 'r') as f:
        return json.load(f)


def run_sahi_experiment(slice_size: int, overlap: float) -> Dict:
    """Run single SAHI experiment with given parameters."""
    print(f"\n{'='*80}")
    print(f"🔬 Running SAHI: slice={slice_size}, overlap={overlap}")
    print(f"{'='*80}\n")
    
    output_file = OUTPUT_DIR / f"sahi_{slice_size}_{int(overlap*10)}.json"
    
    if output_file.exists():
        print(f"✅ Results already exist, loading from {output_file}")
        with open(output_file, 'r') as f:
            return json.load(f)
    
    cmd = [
        "python", "evaluate_visdrone_pipeline.py",
        "--model", MODEL_PATH,
        "--dataset", DATASET_ROOT,
        "--split", SPLIT,
        "--sahi",
        "--slice-size", str(slice_size),
        "--overlap", str(overlap),
        "--imgsz", str(IMGSZ),
        "--conf", str(CONF_THRESHOLD),
        "--iou", str(IOU_THRESHOLD),
        "--device", DEVICE,
        "--output", str(output_file)
    ]
    
    # Run with error handling
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True)
            break
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Experiment failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("🔄 Clearing GPU memory and retrying...")
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                time.sleep(5)  # Wait 5 seconds before retry
            else:
                print("❌ Experiment failed after all retries. Skipping...")
                return None
    
    # Clear GPU memory after successful run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Small delay to ensure GPU is ready for next experiment
    time.sleep(2)
    
    with open(output_file, 'r') as f:
        return json.load(f)


def run_grid_search() -> Dict[str, Dict]:
    """Run complete grid search over all parameter combinations."""
    results = {}
    progress_file = OUTPUT_DIR / "progress.json"
    
    # Load existing progress if available
    if progress_file.exists():
        print(f"📂 Loading existing progress from {progress_file}")
        with open(progress_file, 'r') as f:
            results = json.load(f)
    
    # Run baseline
    if 'baseline' not in results:
        baseline_result = run_baseline()
        results['baseline'] = baseline_result
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(f"✅ Baseline already completed, skipping...")
    
    # Run SAHI experiments
    total_experiments = len(SLICE_SIZES) * len(OVERLAP_RATIOS)
    current = 0
    
    for slice_size in SLICE_SIZES:
        for overlap in OVERLAP_RATIOS:
            current += 1
            print(f"\n{'='*80}")
            print(f"Progress: {current}/{total_experiments} experiments")
            print(f"{'='*80}")
            
            key = f"sahi_{slice_size}_{overlap}"
            
            # Skip if already done
            if key in results and results[key] is not None:
                print(f"✅ Experiment {key} already completed, skipping...")
                continue
            
            result = run_sahi_experiment(slice_size, overlap)
            
            if result is not None:
                results[key] = result
                # Save progress after each successful experiment
                with open(progress_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Progress saved to {progress_file}")
            else:
                print(f"⚠️  Experiment {key} failed, marking as None")
                results[key] = None
    
    return results


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def extract_metrics_table(results: Dict[str, Dict]) -> Dict:
    """Extract metrics into structured format for analysis."""
    data = {
        'config': [],
        'slice_size': [],
        'overlap': [],
        'mAP50': [],
        'mAP50-95': [],
        'mAPs': [],
        'mAPm': [],
        'mAPl': [],
        'mAP50s': [],
        'mAP50m': [],
        'mAP50l': [],
        'fps': [],
        'inference_time_ms': []
    }
    
    # Add baseline
    baseline = results['baseline']
    data['config'].append('Baseline (No SAHI)')
    data['slice_size'].append(0)
    data['overlap'].append(0)
    data['mAP50'].append(baseline['metrics']['mAP50'])
    data['mAP50-95'].append(baseline['metrics']['mAP50-95'])
    data['mAPs'].append(baseline['metrics']['mAPs'])
    data['mAPm'].append(baseline['metrics']['mAPm'])
    data['mAPl'].append(baseline['metrics']['mAPl'])
    data['mAP50s'].append(baseline['metrics']['mAP50s'])
    data['mAP50m'].append(baseline['metrics']['mAP50m'])
    data['mAP50l'].append(baseline['metrics']['mAP50l'])
    data['fps'].append(baseline['metrics']['fps'])
    data['inference_time_ms'].append(baseline['metrics']['avg_inference_time_ms'])
    
    # Add SAHI results
    for key, result in results.items():
        if key == 'baseline':
            continue
        
        # Skip failed experiments
        if result is None:
            print(f"⚠️  Skipping failed experiment: {key}")
            continue
        
        # Parse config from key
        parts = key.split('_')
        slice_size = int(parts[1])
        overlap = float(parts[2])
        
        data['config'].append(f'SAHI {slice_size}/{overlap}')
        data['slice_size'].append(slice_size)
        data['overlap'].append(overlap)
        data['mAP50'].append(result['metrics']['mAP50'])
        data['mAP50-95'].append(result['metrics']['mAP50-95'])
        data['mAPs'].append(result['metrics']['mAPs'])
        data['mAPm'].append(result['metrics']['mAPm'])
        data['mAPl'].append(result['metrics']['mAPl'])
        data['mAP50s'].append(result['metrics']['mAP50s'])
        data['mAP50m'].append(result['metrics']['mAP50m'])
        data['mAP50l'].append(result['metrics']['mAP50l'])
        data['fps'].append(result['metrics']['fps'])
        data['inference_time_ms'].append(result['metrics']['avg_inference_time_ms'])
    
    return data


# ============================================================================
# VISUALIZATION FOR PAPER
# ============================================================================

def create_publication_figures(data: Dict, results: Dict[str, Dict]):
    """Create comprehensive publication-ready figures."""
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 10))
    
    # ========================================================================
    # Subplot 1: Heatmap - mAP50 vs Slice Size vs Overlap
    # ========================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Prepare data for heatmap
    sahi_indices = [i for i, s in enumerate(data['slice_size']) if s > 0]
    slice_sizes_sahi = [data['slice_size'][i] for i in sahi_indices]
    overlaps_sahi = [data['overlap'][i] for i in sahi_indices]
    mAP50_sahi = [data['mAP50'][i] for i in sahi_indices]
    
    # Create pivot table
    unique_slices = sorted(set(slice_sizes_sahi))
    unique_overlaps = sorted(set(overlaps_sahi))
    
    heatmap_data = np.zeros((len(unique_overlaps), len(unique_slices)))
    for i, overlap in enumerate(unique_overlaps):
        for j, slice_size in enumerate(unique_slices):
            idx = [(k, s, o, m) for k, (s, o, m) in enumerate(zip(slice_sizes_sahi, overlaps_sahi, mAP50_sahi)) 
                   if s == slice_size and o == overlap]
            if idx:
                heatmap_data[i, j] = idx[0][3]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=unique_slices, yticklabels=unique_overlaps,
                cbar_kws={'label': 'mAP50'}, ax=ax1)
    ax1.set_xlabel('Slice Size (pixels)')
    ax1.set_ylabel('Overlap Ratio')
    ax1.set_title('(A) mAP50 Heatmap')
    
    # ========================================================================
    # Subplot 2: Line Plot - mAP50 by Slice Size for each Overlap
    # ========================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    # Add baseline
    baseline_mAP50 = data['mAP50'][0]
    ax2.axhline(y=baseline_mAP50, color='black', linestyle='--', linewidth=2, 
                label='Baseline (No SAHI)', alpha=0.7)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_overlaps)))
    for i, overlap in enumerate(unique_overlaps):
        indices = [k for k, (s, o) in enumerate(zip(slice_sizes_sahi, overlaps_sahi)) if o == overlap]
        slices = [slice_sizes_sahi[k] for k in indices]
        mAPs = [mAP50_sahi[k] for k in indices]
        
        # Sort by slice size
        sorted_data = sorted(zip(slices, mAPs))
        slices, mAPs = zip(*sorted_data)
        
        ax2.plot(slices, mAPs, marker='o', linewidth=2, markersize=6, 
                label=f'Overlap={overlap}', color=colors[i])
    
    ax2.set_xlabel('Slice Size (pixels)')
    ax2.set_ylabel('mAP50')
    ax2.set_title('(B) mAP50 vs Slice Size')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # Subplot 3: Scatter Plot - Speed-Accuracy Tradeoff
    # ========================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    # Baseline
    ax3.scatter(data['fps'][0], data['mAP50'][0], s=200, marker='*', 
                c='black', edgecolors='black', linewidth=2, 
                label='Baseline', zorder=10)
    
    # SAHI configurations - color by overlap
    for i, overlap in enumerate(unique_overlaps):
        indices = [k for k in sahi_indices if data['overlap'][k] == overlap]
        fps_vals = [data['fps'][k] for k in indices]
        mAP50_vals = [data['mAP50'][k] for k in indices]
        slice_vals = [data['slice_size'][k] for k in indices]
        
        scatter = ax3.scatter(fps_vals, mAP50_vals, s=100, marker='o', 
                             c=slice_vals, cmap='viridis', alpha=0.7,
                             edgecolors='black', linewidth=0.5)
        
        # Add text labels for slice sizes
        for fps, mAP, slc in zip(fps_vals, mAP50_vals, slice_vals):
            if overlap == 0.2:  # Only label overlap=0.2 to avoid clutter
                ax3.annotate(str(slc), (fps, mAP), fontsize=7, 
                            xytext=(3, 3), textcoords='offset points')
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Slice Size')
    ax3.set_xlabel('FPS (frames per second)')
    ax3.set_ylabel('mAP50')
    ax3.set_title('(C) Speed-Accuracy Tradeoff')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    # ========================================================================
    # Subplot 4: Bar Chart - mAP by Object Size (Best Configs)
    # ========================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    # Find best configurations
    best_accuracy_idx = np.argmax([data['mAP50'][i] for i in sahi_indices])
    best_accuracy_idx = sahi_indices[best_accuracy_idx]
    
    # Find best balanced (highest mAP50 with FPS > 5)
    candidates = [(i, data['mAP50'][i], data['fps'][i]) for i in sahi_indices if data['fps'][i] > 5]
    if candidates:
        best_balanced_idx = max(candidates, key=lambda x: x[1])[0]
    else:
        best_balanced_idx = best_accuracy_idx
    
    # Prepare data
    configs_to_plot = [0, best_accuracy_idx, best_balanced_idx]
    config_names = [
        'Baseline',
        f'Best Acc\n({data["slice_size"][best_accuracy_idx]}/{data["overlap"][best_accuracy_idx]})',
        f'Best Bal\n({data["slice_size"][best_balanced_idx]}/{data["overlap"][best_balanced_idx]})'
    ]
    
    x = np.arange(len(configs_to_plot))
    width = 0.25
    
    mAPs_vals = [data['mAPs'][i] for i in configs_to_plot]
    mAPm_vals = [data['mAPm'][i] for i in configs_to_plot]
    mAPl_vals = [data['mAPl'][i] for i in configs_to_plot]
    
    ax4.bar(x - width, mAPs_vals, width, label='Small', color='#FF6B6B')
    ax4.bar(x, mAPm_vals, width, label='Medium', color='#4ECDC4')
    ax4.bar(x + width, mAPl_vals, width, label='Large', color='#45B7D1')
    
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('mAP50-95')
    ax4.set_title('(D) Performance by Object Size')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names, fontsize=8)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    # ========================================================================
    # Subplot 5: Line Plot - mAPs (Small Objects) Focus
    # ========================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    # Add baseline
    baseline_mAPs = data['mAPs'][0]
    ax5.axhline(y=baseline_mAPs, color='black', linestyle='--', linewidth=2, 
                label='Baseline', alpha=0.7)
    
    for i, overlap in enumerate(unique_overlaps):
        indices = [k for k, (s, o) in enumerate(zip(slice_sizes_sahi, overlaps_sahi)) if o == overlap]
        slices = [slice_sizes_sahi[k] for k in indices]
        mAPs_vals = [data['mAPs'][sahi_indices[k]] for k in range(len(sahi_indices)) 
                     if slice_sizes_sahi[k] in slices and overlaps_sahi[k] == overlap]
        slices_sorted = sorted(set([slice_sizes_sahi[k] for k in range(len(sahi_indices)) 
                                     if overlaps_sahi[k] == overlap]))
        
        # Get corresponding mAPs values
        mAPs_sorted = []
        for slc in slices_sorted:
            idx = [k for k in range(len(sahi_indices)) 
                   if slice_sizes_sahi[k] == slc and overlaps_sahi[k] == overlap]
            if idx:
                mAPs_sorted.append(data['mAPs'][sahi_indices[idx[0]]])
        
        ax5.plot(slices_sorted, mAPs_sorted, marker='o', linewidth=2, markersize=6, 
                label=f'Overlap={overlap}', color=colors[i])
    
    ax5.set_xlabel('Slice Size (pixels)')
    ax5.set_ylabel('mAP50-95 (Small Objects)')
    ax5.set_title('(E) Small Object Detection Performance')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # Subplot 6: Summary Table
    # ========================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    table_data = [
        ['Configuration', 'mAP50', 'mAPs', 'FPS'],
        ['Baseline', f"{data['mAP50'][0]:.3f}", f"{data['mAPs'][0]:.3f}", f"{data['fps'][0]:.1f}"],
        [f"Best Accuracy\n{data['slice_size'][best_accuracy_idx]}/{data['overlap'][best_accuracy_idx]}", 
         f"{data['mAP50'][best_accuracy_idx]:.3f}", 
         f"{data['mAPs'][best_accuracy_idx]:.3f}", 
         f"{data['fps'][best_accuracy_idx]:.1f}"],
        [f"Best Balanced\n{data['slice_size'][best_balanced_idx]}/{data['overlap'][best_balanced_idx]}", 
         f"{data['mAP50'][best_balanced_idx]:.3f}", 
         f"{data['mAPs'][best_balanced_idx]:.3f}", 
         f"{data['fps'][best_balanced_idx]:.1f}"]
    ]
    
    # Calculate improvements
    acc_improve_mAP50 = ((data['mAP50'][best_accuracy_idx] / data['mAP50'][0]) - 1) * 100
    acc_improve_mAPs = ((data['mAPs'][best_accuracy_idx] / data['mAPs'][0]) - 1) * 100 if data['mAPs'][0] > 0 else 0
    
    bal_improve_mAP50 = ((data['mAP50'][best_balanced_idx] / data['mAP50'][0]) - 1) * 100
    bal_improve_mAPs = ((data['mAPs'][best_balanced_idx] / data['mAPs'][0]) - 1) * 100 if data['mAPs'][0] > 0 else 0
    
    table_data.append(['', '', '', ''])
    table_data.append(['Improvements', 'ΔmAP50', 'ΔmAPs', 'ΔFPS'])
    table_data.append(['Best Accuracy', f"+{acc_improve_mAP50:.0f}%", f"+{acc_improve_mAPs:.0f}%", 
                       f"{((data['fps'][best_accuracy_idx] / data['fps'][0]) - 1) * 100:.0f}%"])
    table_data.append(['Best Balanced', f"+{bal_improve_mAP50:.0f}%", f"+{bal_improve_mAPs:.0f}%", 
                       f"{((data['fps'][best_balanced_idx] / data['fps'][0]) - 1) * 100:.0f}%"])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(5, i)].set_facecolor('#FFD93D')
        table[(5, i)].set_text_props(weight='bold')
    
    # Style data rows
    for i in range(1, 5):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    for i in range(6, 8):
        for j in range(4):
            table[(i, j)].set_facecolor('#E8F5E9')
    
    ax6.set_title('(F) Summary Table', pad=20, fontsize=12, weight='bold')
    
    # ========================================================================
    # Save figure
    # ========================================================================
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'grid_search_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")
    plt.close()
    
    # ========================================================================
    # Create simplified figure for paper (2x2 layout)
    # ========================================================================
    fig_paper = plt.figure(figsize=(12, 8))
    
    # Use same plots but cleaner layout
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=unique_slices, yticklabels=unique_overlaps,
                cbar_kws={'label': 'mAP50'}, ax=ax1)
    ax1.set_xlabel('Slice Size (pixels)')
    ax1.set_ylabel('Overlap Ratio')
    ax1.set_title('(A) mAP50 Heatmap: Hyperparameter Grid')
    
    ax2 = plt.subplot(2, 2, 2)
    baseline_mAP50 = data['mAP50'][0]
    ax2.axhline(y=baseline_mAP50, color='black', linestyle='--', linewidth=2, 
                label='Baseline (No SAHI)', alpha=0.7)
    for i, overlap in enumerate(unique_overlaps):
        indices = [k for k, (s, o) in enumerate(zip(slice_sizes_sahi, overlaps_sahi)) if o == overlap]
        slices = [slice_sizes_sahi[k] for k in indices]
        mAPs = [mAP50_sahi[k] for k in indices]
        sorted_data = sorted(zip(slices, mAPs))
        slices, mAPs = zip(*sorted_data)
        ax2.plot(slices, mAPs, marker='o', linewidth=2, markersize=7, 
                label=f'Overlap={overlap}', color=colors[i])
    ax2.set_xlabel('Slice Size (pixels)')
    ax2.set_ylabel('mAP50')
    ax2.set_title('(B) Overall Detection Performance')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(data['fps'][0], data['mAP50'][0], s=300, marker='*', 
                c='gold', edgecolors='black', linewidth=2, 
                label='Baseline (No SAHI)', zorder=10)
    for i, overlap in enumerate(unique_overlaps):
        indices = [k for k in sahi_indices if data['overlap'][k] == overlap]
        fps_vals = [data['fps'][k] for k in indices]
        mAP50_vals = [data['mAP50'][k] for k in indices]
        ax3.scatter(fps_vals, mAP50_vals, s=120, alpha=0.7, 
                   label=f'Overlap={overlap}', color=colors[i],
                   edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('FPS (frames per second)')
    ax3.set_ylabel('mAP50')
    ax3.set_title('(C) Speed-Accuracy Tradeoff')
    ax3.set_xscale('log')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')
    
    ax4 = plt.subplot(2, 2, 4)
    x = np.arange(len(configs_to_plot))
    width = 0.25
    mAPs_vals = [data['mAPs'][i] for i in configs_to_plot]
    mAPm_vals = [data['mAPm'][i] for i in configs_to_plot]
    mAPl_vals = [data['mAPl'][i] for i in configs_to_plot]
    ax4.bar(x - width, mAPs_vals, width, label='Small (<32²px)', color='#FF6B6B', edgecolor='black', linewidth=0.5)
    ax4.bar(x, mAPm_vals, width, label='Medium (32²-96²px)', color='#4ECDC4', edgecolor='black', linewidth=0.5)
    ax4.bar(x + width, mAPl_vals, width, label='Large (>96²px)', color='#45B7D1', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('mAP50-95')
    ax4.set_title('(D) Performance by Object Size')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names, fontsize=9)
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path_paper = OUTPUT_DIR / 'grid_search_paper_figure.png'
    plt.savefig(output_path_paper, dpi=300, bbox_inches='tight')
    print(f"✅ Paper figure saved to: {output_path_paper}")
    plt.close()
    
    return best_accuracy_idx, best_balanced_idx


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("🔬 SAHI Hyperparameter Grid Search for VisDrone")
    print(f"{'='*80}")
    print(f"\n📋 Configuration:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Dataset: {DATASET_ROOT}")
    print(f"   Split: {SPLIT}")
    print(f"   Image size (imgsz): {IMGSZ}")
    print(f"   Confidence threshold: {CONF_THRESHOLD}")
    print(f"   IoU threshold: {IOU_THRESHOLD}")
    print(f"   Device: {DEVICE}")
    print(f"\n🔍 Hyperparameter Grid:")
    print(f"   Slice sizes: {SLICE_SIZES}")
    print(f"   Overlap ratios: {OVERLAP_RATIOS}")
    print(f"   Total experiments: 1 baseline + {len(SLICE_SIZES) * len(OVERLAP_RATIOS)} SAHI configs")
    print(f"\n📁 Results will be saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    # Confirm before starting
    response = input("Press ENTER to start grid search (or Ctrl+C to cancel)... ")
    
    # Run grid search
    start_time = datetime.now()
    results = run_grid_search()
    end_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"✅ Grid search completed in {end_time - start_time}")
    print(f"{'='*80}\n")
    
    # Save all results to single file
    results_file = OUTPUT_DIR / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 All results saved to: {results_file}\n")
    
    # Extract metrics and create visualizations
    print("📊 Analyzing results and creating visualizations...\n")
    data = extract_metrics_table(results)
    best_acc_idx, best_bal_idx = create_publication_figures(data, results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 GRID SEARCH SUMMARY")
    print(f"{'='*80}")
    print(f"\n🏆 Best Configurations:")
    print(f"\n1. Best Accuracy:")
    print(f"   Slice: {data['slice_size'][best_acc_idx]}, Overlap: {data['overlap'][best_acc_idx]}")
    print(f"   mAP50: {data['mAP50'][best_acc_idx]:.4f} (+{((data['mAP50'][best_acc_idx]/data['mAP50'][0])-1)*100:.0f}%)")
    print(f"   mAPs:  {data['mAPs'][best_acc_idx]:.4f} (+{((data['mAPs'][best_acc_idx]/data['mAPs'][0])-1)*100:.0f}% if data['mAPs'][0] > 0 else 'N/A')")
    print(f"   FPS:   {data['fps'][best_acc_idx]:.2f}")
    
    print(f"\n2. Best Balanced (FPS > 5):")
    print(f"   Slice: {data['slice_size'][best_bal_idx]}, Overlap: {data['overlap'][best_bal_idx]}")
    print(f"   mAP50: {data['mAP50'][best_bal_idx]:.4f} (+{((data['mAP50'][best_bal_idx]/data['mAP50'][0])-1)*100:.0f}%)")
    print(f"   mAPs:  {data['mAPs'][best_bal_idx]:.4f} (+{((data['mAPs'][best_bal_idx]/data['mAPs'][0])-1)*100:.0f}% if data['mAPs'][0] > 0 else 'N/A')")
    print(f"   FPS:   {data['fps'][best_bal_idx]:.2f}")
    
    print(f"\n3. Baseline:")
    print(f"   mAP50: {data['mAP50'][0]:.4f}")
    print(f"   mAPs:  {data['mAPs'][0]:.4f}")
    print(f"   FPS:   {data['fps'][0]:.2f}")
    
    print(f"\n{'='*80}")
    print(f"✅ All analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

