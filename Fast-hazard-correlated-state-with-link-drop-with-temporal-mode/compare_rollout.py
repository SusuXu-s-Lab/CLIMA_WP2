#!/usr/bin/env python3
# compare_rollout.py
# Aggregate and visualize rollout test metrics across three time modes
# Focuses on cascade-level metrics: cumulative RMSE, correlation, final size error, etc.

import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

SWEEP_DIR = "sweep_results_rollout"


def aggregate_seeds(pattern):
    """Aggregate metrics across seeds. Returns dict[config] -> {mean, std, n_seeds}."""
    data_by_config = defaultdict(list)
    
    for path in sorted(glob.glob(pattern)):
        config_name = re.sub(r'_seed\d+$', '', Path(path).parent.name)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Extract test metrics (we want the 'test' section)
                if 'test' in data:
                    data_by_config[config_name].append(data['test'])
                else:
                    print(f"Warning: No 'test' section in {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    
    aggregated = {}
    for config, metrics_list in data_by_config.items():
        if not metrics_list:
            continue
        mean_metrics, std_metrics = {}, {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
                if vals:
                    # Use nanmean/nanstd to ignore NaN values from individual seeds
                    mean_metrics[key] = np.nanmean(vals)
                    std_metrics[key] = np.nanstd(vals)
        aggregated[config] = {'mean': mean_metrics, 'std': std_metrics, 'n_seeds': len(metrics_list)}
    
    return aggregated


def plot_rollout_comparison(modes=['warmup', 'inverse_u', 'inverse_u_with_time'], output_dir=None):
    """Compare rollout test metrics across modes."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for all modes
    mode_data = {}
    for mode in modes:
        agg = aggregate_seeds(os.path.join(SWEEP_DIR, mode, "speed_*/metrics_split.json"))
        if agg:
            mode_data[mode] = agg
        else:
            print(f"Warning: No data found for mode '{mode}'")
    
    if not mode_data:
        print("No data found for any mode!")
        return
    
    # Define test metrics to plot (per dimension)
    D = 3  # Number of dimensions
    
    # Speed order
    speed_order = ['very_slow', 'slow', 'medium', 'fast']
    x_pos = np.arange(len(speed_order))
    
    # Plot 1: Cumulative RMSE per dimension
    fig, axes = plt.subplots(1, D, figsize=(15, 4.5))
    if D == 1:
        axes = [axes]
    
    for d, ax in enumerate(axes):
        for mode_name, mode_color, mode_marker in zip(
            modes,
            ['#1f77b4', '#ff7f0e', '#2ca02c'],  # blue, orange, green
            ['o', 's', '^']
        ):
            if mode_name not in mode_data:
                continue
            
            configs = [f'speed_{s}' for s in speed_order if f'speed_{s}' in mode_data[mode_name]]
            metric_key = f'test_cumulative_rmse_d{d}'
            
            y = [mode_data[mode_name][c]['mean'].get(metric_key, np.nan) for c in configs]
            e = [mode_data[mode_name][c]['std'].get(metric_key, 0) for c in configs]
            
            ax.errorbar(x_pos[:len(y)], y, yerr=e, 
                       fmt=f'{mode_marker}-', label=mode_name.replace('_', ' ').title(),
                       linewidth=2, capsize=5, markersize=7, color=mode_color)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in speed_order], fontsize=10)
        ax.set_xlabel('Diffusion Speed', fontsize=11)
        ax.set_ylabel('Cumulative RMSE', fontsize=11)
        ax.set_title(f'Dimension {d}: Cumulative Activation RMSE', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rollout_cumulative_rmse_by_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'rollout_cumulative_rmse_by_dim.png')}")
    
    # Plot 2: Cumulative Correlation per dimension
    fig, axes = plt.subplots(1, D, figsize=(15, 4.5))
    if D == 1:
        axes = [axes]
    
    for d, ax in enumerate(axes):
        for mode_name, mode_color, mode_marker in zip(
            modes,
            ['#1f77b4', '#ff7f0e', '#2ca02c'],
            ['o', 's', '^']
        ):
            if mode_name not in mode_data:
                continue
            
            configs = [f'speed_{s}' for s in speed_order if f'speed_{s}' in mode_data[mode_name]]
            metric_key = f'test_cumulative_corr_d{d}'
            
            y = [mode_data[mode_name][c]['mean'].get(metric_key, np.nan) for c in configs]
            e = [mode_data[mode_name][c]['std'].get(metric_key, 0) for c in configs]
            
            ax.errorbar(x_pos[:len(y)], y, yerr=e,
                       fmt=f'{mode_marker}-', label=mode_name.replace('_', ' ').title(),
                       linewidth=2, capsize=5, markersize=7, color=mode_color)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect correlation')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in speed_order], fontsize=10)
        ax.set_xlabel('Diffusion Speed', fontsize=11)
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_title(f'Dimension {d}: Cumulative Activation Correlation', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rollout_cumulative_corr_by_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'rollout_cumulative_corr_by_dim.png')}")
    
    # Plot 3: Final Size Error per dimension
    fig, axes = plt.subplots(1, D, figsize=(15, 4.5))
    if D == 1:
        axes = [axes]
    
    for d, ax in enumerate(axes):
        for mode_name, mode_color, mode_marker in zip(
            modes,
            ['#1f77b4', '#ff7f0e', '#2ca02c'],
            ['o', 's', '^']
        ):
            if mode_name not in mode_data:
                continue
            
            configs = [f'speed_{s}' for s in speed_order if f'speed_{s}' in mode_data[mode_name]]
            metric_key = f'test_final_size_error_d{d}'
            
            y = [mode_data[mode_name][c]['mean'].get(metric_key, np.nan) for c in configs]
            e = [mode_data[mode_name][c]['std'].get(metric_key, 0) for c in configs]
            
            ax.errorbar(x_pos[:len(y)], y, yerr=e,
                       fmt=f'{mode_marker}-', label=mode_name.replace('_', ' ').title(),
                       linewidth=2, capsize=5, markersize=7, color=mode_color)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect prediction')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in speed_order], fontsize=10)
        ax.set_xlabel('Diffusion Speed', fontsize=11)
        ax.set_ylabel('Final Size Error (|GT - RO|)', fontsize=11)
        ax.set_title(f'Dimension {d}: Final Cascade Size Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rollout_final_size_error_by_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'rollout_final_size_error_by_dim.png')}")
    
    # Plot 4: First Activation MAE per dimension
    fig, axes = plt.subplots(1, D, figsize=(15, 4.5))
    if D == 1:
        axes = [axes]
    
    for d, ax in enumerate(axes):
        for mode_name, mode_color, mode_marker in zip(
            modes,
            ['#1f77b4', '#ff7f0e', '#2ca02c'],
            ['o', 's', '^']
        ):
            if mode_name not in mode_data:
                continue
            
            configs = [f'speed_{s}' for s in speed_order if f'speed_{s}' in mode_data[mode_name]]
            metric_key = f'test_first_activation_mae_d{d}'
            
            y = [mode_data[mode_name][c]['mean'].get(metric_key, np.nan) for c in configs]
            e = [mode_data[mode_name][c]['std'].get(metric_key, 0) for c in configs]
            
            ax.errorbar(x_pos[:len(y)], y, yerr=e,
                       fmt=f'{mode_marker}-', label=mode_name.replace('_', ' ').title(),
                       linewidth=2, capsize=5, markersize=7, color=mode_color)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect timing')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in speed_order], fontsize=10)
        ax.set_xlabel('Diffusion Speed', fontsize=11)
        ax.set_ylabel('First Activation MAE (steps)', fontsize=11)
        ax.set_title(f'Dimension {d}: First Activation Time Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rollout_first_activation_mae_by_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'rollout_first_activation_mae_by_dim.png')}")


def generate_summary_tables(modes=['warmup', 'inverse_u', 'inverse_u_with_time'], output_dir=None):
    """Generate summary tables for rollout test metrics."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    D = 3  # Number of dimensions
    
    with open(os.path.join(output_dir, 'rollout_summary.txt'), 'w') as f:
        f.write("ROLLOUT TEST METRICS SUMMARY (mean ± std across seeds)\n")
        f.write("="*180 + "\n\n")
        
        for mode in modes:
            agg = aggregate_seeds(os.path.join(SWEEP_DIR, mode, "speed_*/metrics_split.json"))
            if not agg:
                continue
            
            f.write(f"\n{mode.upper().replace('_', ' ')}\n")
            f.write("-"*180 + "\n\n")
            
            # Cumulative RMSE
            f.write("CUMULATIVE ACTIVATION RMSE (per dimension)\n")
            header = f"{'Config':<20}"
            for d in range(D):
                header += f" {'Dim'+str(d):<18}"
            f.write(header + "\n")
            f.write("-"*100 + "\n")
            
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                row = f"{config:<20}"
                for d in range(D):
                    key = f'test_cumulative_rmse_d{d}'
                    row += f" {mean.get(key,0):.3f}±{std.get(key,0):.3f}    "
                row += f"(n={n})"
                f.write(row + "\n")
            
            # Cumulative Correlation
            f.write("\n\nCUMULATIVE ACTIVATION CORRELATION (per dimension)\n")
            header = f"{'Config':<20}"
            for d in range(D):
                header += f" {'Dim'+str(d):<18}"
            f.write(header + "\n")
            f.write("-"*100 + "\n")
            
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                row = f"{config:<20}"
                for d in range(D):
                    key = f'test_cumulative_corr_d{d}'
                    row += f" {mean.get(key,0):.3f}±{std.get(key,0):.3f}    "
                row += f"(n={n})"
                f.write(row + "\n")
            
            # Final Size Error
            f.write("\n\nFINAL CASCADE SIZE ERROR (per dimension)\n")
            header = f"{'Config':<20}"
            for d in range(D):
                header += f" {'Dim'+str(d):<18}"
            f.write(header + "\n")
            f.write("-"*100 + "\n")
            
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                row = f"{config:<20}"
                for d in range(D):
                    key = f'test_final_size_error_d{d}'
                    row += f" {mean.get(key,0):.1f}±{std.get(key,0):.1f}      "
                row += f"(n={n})"
                f.write(row + "\n")
            
            # First Activation MAE
            f.write("\n\nFIRST ACTIVATION TIME MAE (per dimension, in steps)\n")
            header = f"{'Config':<20}"
            for d in range(D):
                header += f" {'Dim'+str(d):<18}"
            f.write(header + "\n")
            f.write("-"*100 + "\n")
            
            for config in sorted(agg.keys()):
                mean = agg[config]['mean']
                std = agg[config]['std']
                n = agg[config]['n_seeds']
                row = f"{config:<20}"
                for d in range(D):
                    key = f'test_first_activation_mae_d{d}'
                    row += f" {mean.get(key,0):.2f}±{std.get(key,0):.2f}    "
                row += f"(n={n})"
                f.write(row + "\n")
            
            f.write("\n" + "="*180 + "\n")
    
    print(f"Saved: {os.path.join(output_dir, 'rollout_summary.txt')}")


def main():
    print("Generating rollout comparison plots and summaries...")
    
    modes = ['warmup', 'inverse_u', 'inverse_u_with_time']
    
    plot_rollout_comparison(modes=modes)
    generate_summary_tables(modes=modes)
    
    print("\nRollout comparison analysis completed!")
    print(f"Check {SWEEP_DIR}/comparison_plots/ for outputs")


if __name__ == '__main__':
    main()
