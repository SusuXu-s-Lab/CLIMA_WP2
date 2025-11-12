#!/usr/bin/env python3
# compare_results_v2.py
# Aggregate and visualize metrics from sweep experiments with multi-seed support
# Compares warmup vs inverse_u time modes

import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

SWEEP_DIR = "sweep_results"


def aggregate_seeds(pattern):
    """Aggregate metrics across seeds. Returns dict[config] -> {mean, std, n_seeds}."""
    data_by_config = defaultdict(list)
    
    for path in sorted(glob.glob(pattern)):
        config_name = re.sub(r'_seed\d+$', '', Path(path).parent.name)
        with open(path, 'r') as f:
            data_by_config[config_name].append(json.load(f))
    
    aggregated = {}
    for config, metrics_list in data_by_config.items():
        mean_metrics, std_metrics = {}, {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
                if vals:
                    mean_metrics[key] = np.mean(vals)
                    std_metrics[key] = np.std(vals)
        aggregated[config] = {'mean': mean_metrics, 'std': std_metrics, 'n_seeds': len(metrics_list)}
    
    return aggregated


def plot_mode_comparison(mode1='warmup', mode2='inverse_u', mode3=None, output_dir=None):
    """Compare two or three time modes on key metrics."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics by category
    metrics = [
        # Mechanism fidelity
        ('Self_R2', 'Self R² (Mechanism)', True),
        ('B_R2', 'Neighbor R² (Mechanism)', True),
        ('Attr_R2_vsTruePosterior', 'Attribution R² (Mechanism)', True),
        # Event probabilistic
        ('NLL', 'NLL (Event Probabilistic)', False),
        ('Brier', 'Brier Score (Event Probabilistic)', False),
        # Calibration
        ('ECE', 'ECE (Calibration)', False),
        ('ECE_adaptive', 'Adaptive ECE (Calibration)', False),
        # Discrimination
        ('ROC_AUC', 'ROC-AUC (Discrimination)', True),
        ('PR_AUC', 'PR-AUC (Discrimination)', True),
    ]
    
    # Density comparison
    agg1_density = aggregate_seeds(os.path.join(SWEEP_DIR, mode1, "density_*/metrics.json"))
    agg2_density = aggregate_seeds(os.path.join(SWEEP_DIR, mode2, "density_*/metrics.json"))
    agg3_density = aggregate_seeds(os.path.join(SWEEP_DIR, mode3, "density_*/metrics.json")) if mode3 else None
    
    if agg1_density and agg2_density:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()
        
        for ax, (metric_key, metric_label, higher_better) in zip(axes, metrics):
            configs1 = sorted([k for k in agg1_density.keys() if k.startswith('density_')])
            x_vals = [float(c.split('_')[1]) for c in configs1]
            
            y1 = [agg1_density[c]['mean'][metric_key] for c in configs1]
            e1 = [agg1_density[c]['std'][metric_key] for c in configs1]
            
            configs2 = sorted([k for k in agg2_density.keys() if k.startswith('density_')])
            y2 = [agg2_density[c]['mean'][metric_key] for c in configs2]
            e2 = [agg2_density[c]['std'][metric_key] for c in configs2]
            
            ax.errorbar(x_vals, y1, yerr=e1, fmt='o-', label=mode1, linewidth=2, capsize=5, markersize=6)
            ax.errorbar(x_vals, y2, yerr=e2, fmt='s--', label=mode2, linewidth=2, capsize=5, markersize=6)
            
            # Add third mode if provided
            if agg3_density:
                configs3 = sorted([k for k in agg3_density.keys() if k.startswith('density_')])
                y3 = [agg3_density[c]['mean'][metric_key] for c in configs3]
                e3 = [agg3_density[c]['std'][metric_key] for c in configs3]
                ax.errorbar(x_vals, y3, yerr=e3, fmt='^:', label=mode3, linewidth=2, capsize=5, markersize=6)
            
            # Add y=0 reference line and failure zone for R² metrics
            if 'R2' in metric_key or 'R²' in metric_label:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
                # Mark failure zone (R² < 0) with red shading
                all_vals = y1 + y2
                if agg3_density:
                    all_vals += y3
                min_val = min(all_vals)
                if min_val < 0:
                    ax.axhspan(min(min_val * 1.1, -1.0), 0, alpha=0.1, color='red', zorder=0)
                    ax.set_ylim(bottom=max(min_val * 1.1, -1.0), top=1.0)  # Set upper limit to 1.0
                else:
                    ax.set_ylim(top=1.0)  # Set upper limit to 1.0 even when no negative values
            
            ax.set_xlabel('Network Density (p_edge)', fontsize=10)
            ax.set_ylabel(metric_label.split('(')[0].strip(), fontsize=10)
            ax.set_title(metric_label, fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename_suffix = f'{mode1}_vs_{mode2}' + (f'_vs_{mode3}' if mode3 else '')
        plt.savefig(os.path.join(output_dir, f'density_{filename_suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, f'density_{filename_suffix}.png')}")
    
    # Speed comparison (now also line plots)
    agg1_speed = aggregate_seeds(os.path.join(SWEEP_DIR, mode1, "speed_*/metrics.json"))
    agg2_speed = aggregate_seeds(os.path.join(SWEEP_DIR, mode2, "speed_*/metrics.json"))
    agg3_speed = aggregate_seeds(os.path.join(SWEEP_DIR, mode3, "speed_*/metrics.json")) if mode3 else None
    
    if agg1_speed and agg2_speed:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()
        
        speed_order = ['very_slow', 'slow', 'medium', 'fast']
        x_pos = np.arange(len(speed_order))
        
        for ax, (metric_key, metric_label, higher_better) in zip(axes, metrics):
            configs1 = [f'speed_{s}' for s in speed_order if f'speed_{s}' in agg1_speed]
            y1 = [agg1_speed[c]['mean'][metric_key] for c in configs1]
            e1 = [agg1_speed[c]['std'][metric_key] for c in configs1]
            
            configs2 = [f'speed_{s}' for s in speed_order if f'speed_{s}' in agg2_speed]
            y2 = [agg2_speed[c]['mean'][metric_key] for c in configs2]
            e2 = [agg2_speed[c]['std'][metric_key] for c in configs2]
            
            # Use line plot instead of bar
            ax.errorbar(x_pos, y1, yerr=e1, fmt='o-', label=mode1, linewidth=2, capsize=5, markersize=6)
            ax.errorbar(x_pos, y2, yerr=e2, fmt='s--', label=mode2, linewidth=2, capsize=5, markersize=6)
            
            # Add third mode if provided
            if agg3_speed:
                configs3 = [f'speed_{s}' for s in speed_order if f'speed_{s}' in agg3_speed]
                y3 = [agg3_speed[c]['mean'][metric_key] for c in configs3]
                e3 = [agg3_speed[c]['std'][metric_key] for c in configs3]
                ax.errorbar(x_pos, y3, yerr=e3, fmt='^:', label=mode3, linewidth=2, capsize=5, markersize=6)
            
            # Add y=0 reference line and failure zone for R² metrics
            if 'R2' in metric_key or 'R²' in metric_label:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
                # Mark failure zone (R² < 0) with red shading
                all_vals = y1 + y2
                if agg3_speed:
                    all_vals += y3
                min_val = min(all_vals)
                if min_val < 0:
                    ax.axhspan(min(min_val * 1.1, -1.0), 0, alpha=0.1, color='red', zorder=0)
                    ax.set_ylim(bottom=max(min_val * 1.1, -1.0), top=1.0)  # Set upper limit to 1.0
                else:
                    ax.set_ylim(top=1.0)  # Set upper limit to 1.0 even when no negative values
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([s.replace('_', ' ').title() for s in speed_order], fontsize=9)
            ax.set_xlabel('Diffusion Speed', fontsize=10)
            ax.set_ylabel(metric_label.split('(')[0].strip(), fontsize=10)
            ax.set_title(metric_label, fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename_suffix = f'{mode1}_vs_{mode2}' + (f'_vs_{mode3}' if mode3 else '')
        plt.savefig(os.path.join(output_dir, f'speed_{filename_suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, f'speed_{filename_suffix}.png')}")


def generate_summary_tables(output_dir=None):
    """Generate summary tables for all modes."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    metric_keys = ['Self_R2', 'B_R2', 'Attr_R2_vsTruePosterior', 
                   'NLL', 'Brier', 
                   'ECE', 'ECE_adaptive',
                   'ROC_AUC', 'PR_AUC']
    
    for mode in ['warmup', 'inverse_u', 'inverse_u_with_time']:
        agg = aggregate_seeds(os.path.join(SWEEP_DIR, mode, "*/metrics.json"))
        if not agg:
            continue
        
        with open(os.path.join(output_dir, f'{mode}_summary.txt'), 'w') as f:
            f.write(f"{mode.upper()} Mode Summary (mean ± std across seeds)\n")
            f.write("="*150 + "\n\n")
            
            # Mechanism fidelity
            f.write("MECHANISM FIDELITY\n")
            f.write(f"{'Config':<25} {'Self_R2':<18} {'Neighbor_R2':<18} {'Attr_R2':<18}\n")
            f.write("-"*80 + "\n")
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                f.write(f"{config:<25} "
                       f"{mean.get('Self_R2',0):.3f}±{std.get('Self_R2',0):.3f}  "
                       f"{mean.get('B_R2',0):.3f}±{std.get('B_R2',0):.3f}      "
                       f"{mean.get('Attr_R2_vsTruePosterior',0):.3f}±{std.get('Attr_R2_vsTruePosterior',0):.3f}  "
                       f"(n={n})\n")
            
            # Event probabilistic
            f.write("\n\nEVENT PROBABILISTIC METRICS\n")
            f.write(f"{'Config':<25} {'NLL':<18} {'Brier':<18}\n")
            f.write("-"*65 + "\n")
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                f.write(f"{config:<25} "
                       f"{mean.get('NLL',0):.3f}±{std.get('NLL',0):.3f}  "
                       f"{mean.get('Brier',0):.3f}±{std.get('Brier',0):.3f}  "
                       f"(n={n})\n")
            
            # Calibration
            f.write("\n\nCALIBRATION METRICS\n")
            f.write(f"{'Config':<25} {'ECE':<18} {'Adaptive_ECE':<18}\n")
            f.write("-"*65 + "\n")
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                f.write(f"{config:<25} "
                       f"{mean.get('ECE',0):.3f}±{std.get('ECE',0):.3f}  "
                       f"{mean.get('ECE_adaptive',0):.3f}±{std.get('ECE_adaptive',0):.3f}  "
                       f"(n={n})\n")
            
            # Discrimination
            f.write("\n\nDISCRIMINATION METRICS\n")
            f.write(f"{'Config':<25} {'ROC-AUC':<18} {'PR-AUC':<18}\n")
            f.write("-"*65 + "\n")
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                f.write(f"{config:<25} "
                       f"{mean.get('ROC_AUC',0):.3f}±{std.get('ROC_AUC',0):.3f}  "
                       f"{mean.get('PR_AUC',0):.3f}±{std.get('PR_AUC',0):.3f}  "
                       f"(n={n})\n")
        
        print(f"Saved: {os.path.join(output_dir, f'{mode}_summary.txt')}")


def main():
    print("Generating comparison plots and summaries...")
    
    # Three-way comparison with all modes
    plot_mode_comparison(mode1='warmup', mode2='inverse_u', mode3='inverse_u_with_time')
    
    generate_summary_tables()
    
    print("\nComparison analysis completed!")


if __name__ == '__main__':
    main()
