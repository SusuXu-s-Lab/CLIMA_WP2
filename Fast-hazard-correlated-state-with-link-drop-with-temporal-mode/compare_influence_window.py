#!/usr/bin/env python3
# compare_influence_window.py
# Analyze and visualize results from influence window (m) experiments

import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

SWEEP_DIR = "sweep_results_m"


def aggregate_seeds(pattern):
    """Aggregate metrics across seeds."""
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
        aggregated[config] = {
            'mean': mean_metrics,
            'std': std_metrics,
            'n_seeds': len(metrics_list)
        }
    
    return aggregated


def plot_matched_m_results(output_dir=None):
    """Plot Phase 1 results: matched m experiments across time modes."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate matched experiments for each time mode
    time_modes = ['warmup', 'inverse_u', 'inverse_u_with_time']
    agg_by_mode = {}
    
    for mode in time_modes:
        agg = aggregate_seeds(os.path.join(SWEEP_DIR, 'matched', mode, 'm*/metrics.json'))
        if agg:
            agg_by_mode[mode] = agg
    
    if not agg_by_mode:
        print("No matched m results found.")
        return
    
    # Extract m values (assume same across modes)
    first_mode = list(agg_by_mode.values())[0]
    configs = sorted([k for k in first_mode.keys() if k.startswith('m')])
    m_values = [int(c[1:]) for c in configs]
    
    # Key metrics to compare (same as compare_results.py)
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
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    # Color scheme for three modes
    colors = {'warmup': '#2E86AB', 'inverse_u': '#A23B72', 'inverse_u_with_time': '#F18F01'}
    markers = {'warmup': 'o', 'inverse_u': 's', 'inverse_u_with_time': '^'}
    labels = {'warmup': 'Warmup', 'inverse_u': 'Inverse-U', 'inverse_u_with_time': 'Inverse-U + Time'}
    
    for ax, (metric_key, metric_label, higher_better) in zip(axes, metrics):
        for mode in time_modes:
            if mode not in agg_by_mode:
                continue
            
            y_mean = [agg_by_mode[mode][c]['mean'][metric_key] for c in configs]
            y_std = [agg_by_mode[mode][c]['std'][metric_key] for c in configs]
            
            ax.errorbar(m_values, y_mean, yerr=y_std, 
                       fmt=f'{markers[mode]}-', linewidth=2,
                       capsize=5, markersize=8, 
                       color=colors[mode], label=labels[mode])
        
        # Add y=0 reference for R² metrics
        if 'R2' in metric_key or 'R²' in metric_label:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Influence Window (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_label.split('(')[0].strip(), fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.set_xticks(m_values)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'matched_m_by_timemode.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_mismatch_results(output_dir=None):
    """Plot Phase 2 results: mismatched m experiments across time modes."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Key metrics (same as compare_results.py)
    metrics = [
        ('Self_R2', 'Self R²'),
        ('B_R2', 'Edge R²'),
        ('Attr_R2_vsTruePosterior', 'Attribution R²'),
        ('NLL', 'NLL'),
        ('Brier', 'Brier'),
        ('ECE', 'ECE'),
        ('ECE_adaptive', 'Adaptive ECE'),
        ('ROC_AUC', 'ROC-AUC'),
        ('PR_AUC', 'PR-AUC'),
    ]
    
    time_modes = ['warmup', 'inverse_u', 'inverse_u_with_time']
    mode_labels = {'warmup': 'Warmup', 'inverse_u': 'Inverse-U', 'inverse_u_with_time': 'Inverse-U + Time'}
    
    # Define mismatch scenarios and their baselines
    scenarios = {
        'true_m3_train_m1': {'label': 'Data m=3\nTrain m=1', 'baseline_m': 'm3'},
        'true_m1_train_m3': {'label': 'Data m=1\nTrain m=3', 'baseline_m': 'm1'},
        'true_m5_train_m2': {'label': 'Data m=5\nTrain m=2', 'baseline_m': 'm5'},
    }
    
    # Generate one plot per time mode
    for mode in time_modes:
        # Aggregate mismatch experiments for this mode
        mismatch_pattern = os.path.join(SWEEP_DIR, 'mismatch', mode, '*/metrics.json')
        agg_mismatch = aggregate_seeds(mismatch_pattern)
        
        # Get matched baselines for this mode
        matched_pattern = os.path.join(SWEEP_DIR, 'matched', mode, '*/metrics.json')
        agg_matched = aggregate_seeds(matched_pattern)
        
        if not agg_mismatch:
            print(f"No mismatch results found for {mode}.")
            continue
        
        if not agg_matched:
            print(f"No matched baselines found for {mode}.")
            continue
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()
        
        for ax, (metric_key, metric_label) in zip(axes, metrics):
            # Three groups, each with baseline vs mismatch
            n_scenarios = len(scenarios)
            x_pos = np.arange(n_scenarios)
            width = 0.35
            
            baseline_means, baseline_stds = [], []
            mismatch_means, mismatch_stds = [], []
            scenario_labels = []
            
            for scenario_key, scenario_info in scenarios.items():
                scenario_labels.append(scenario_info['label'])
                baseline_key = scenario_info['baseline_m']
                
                # Matched baseline
                if baseline_key in agg_matched:
                    baseline_means.append(agg_matched[baseline_key]['mean'][metric_key])
                    baseline_stds.append(agg_matched[baseline_key]['std'][metric_key])
                else:
                    baseline_means.append(np.nan)
                    baseline_stds.append(0)
                
                # Mismatch result
                if scenario_key in agg_mismatch:
                    mismatch_means.append(agg_mismatch[scenario_key]['mean'][metric_key])
                    mismatch_stds.append(agg_mismatch[scenario_key]['std'][metric_key])
                else:
                    mismatch_means.append(np.nan)
                    mismatch_stds.append(0)
            
            # Plot grouped bars
            ax.bar(x_pos - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Matched (baseline)', capsize=5, 
                   color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1)
            ax.bar(x_pos + width/2, mismatch_means, width, yerr=mismatch_stds,
                   label='Mismatched', capsize=5,
                   color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add y=0 reference for R² metrics
            if 'R2' in metric_key or 'R²' in metric_label:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_label}: {mode_labels[mode]}', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scenario_labels, fontsize=9)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = os.path.join(output_dir, f'mismatch_{mode}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


def generate_summary_table(output_dir=None):
    """Generate text summary of all experiments."""
    if output_dir is None:
        output_dir = os.path.join(SWEEP_DIR, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("INFLUENCE WINDOW (m) EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Phase 1: Matched m across time modes
        f.write("PHASE 1: MATCHED m (Data m = Train m) - All Time Modes\n")
        f.write("-"*80 + "\n\n")
        
        time_modes = ['warmup', 'inverse_u', 'inverse_u_with_time']
        for mode in time_modes:
            agg_matched = aggregate_seeds(os.path.join(SWEEP_DIR, 'matched', mode, '*/metrics.json'))
            if not agg_matched:
                continue
            
            f.write(f"\n{mode.upper().replace('_', ' ')}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'m':<8} {'Self_R2':<12} {'Edge_R2':<12} {'Attr_R2':<12} {'NLL':<10} {'Brier':<10}\n")
            f.write("-"*70 + "\n")
            
            for config in sorted([k for k in agg_matched.keys() if k.startswith('m')]):
                mean = agg_matched[config]['mean']
                std = agg_matched[config]['std']
                m_val = config[1:]  # Extract m value
                f.write(f"{m_val:<8} "
                       f"{mean['Self_R2']:.3f}±{std['Self_R2']:.3f}  "
                       f"{mean['B_R2']:.3f}±{std['B_R2']:.3f}  "
                       f"{mean['Attr_R2_vsTruePosterior']:.3f}±{std['Attr_R2_vsTruePosterior']:.3f}  "
                       f"{mean['NLL']:.3f}±{std['NLL']:.3f}  "
                       f"{mean['Brier']:.3f}±{std['Brier']:.3f}\n")
        
        # Phase 2: Mismatch across time modes
        f.write("\n\nPHASE 2: MISMATCHED m (Data m ≠ Train m) - All Time Modes\n")
        f.write("-"*80 + "\n")
        f.write("Three scenarios tested:\n")
        f.write("  1. true_m3_train_m1: Data generated with m=3, model trained with m=1\n")
        f.write("  2. true_m1_train_m3: Data generated with m=1, model trained with m=3\n")
        f.write("  3. true_m5_train_m2: Data generated with m=5, model trained with m=2\n\n")
        
        for mode in time_modes:
            agg_mismatch = aggregate_seeds(os.path.join(SWEEP_DIR, 'mismatch', mode, '*/metrics.json'))
            if not agg_mismatch:
                continue
            
            f.write(f"\n{mode.upper().replace('_', ' ')}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Scenario':<20} {'Self_R2':<12} {'Edge_R2':<12} {'Attr_R2':<12} {'NLL':<10} {'Brier':<10}\n")
            f.write("-"*70 + "\n")
            
            for config in sorted(agg_mismatch.keys()):
                mean = agg_mismatch[config]['mean']
                std = agg_mismatch[config]['std']
                f.write(f"{config:<20} "
                       f"{mean['Self_R2']:.3f}±{std['Self_R2']:.3f}  "
                       f"{mean['B_R2']:.3f}±{std['B_R2']:.3f}  "
                       f"{mean['Attr_R2_vsTruePosterior']:.3f}±{std['Attr_R2_vsTruePosterior']:.3f}  "
                       f"{mean['NLL']:.3f}±{std['NLL']:.3f}  "
                       f"{mean['Brier']:.3f}±{std['Brier']:.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Saved: {os.path.join(output_dir, 'summary.txt')}")


def main():
    print("Analyzing influence window experiments...\n")
    
    plot_matched_m_results()
    plot_mismatch_results()
    generate_summary_table()
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
