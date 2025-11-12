#!/usr/bin/env python3
# sweep_experiments.py
# Batch run experiments to test model capacity under different scenarios:
# (A) Network density (p_edge)
# (B) Diffusion speed (c0, c1, a0)
# Supports multiple random seeds and two time modulation modes (warmup / inverse_u)

import subprocess
import sys
import os
from pathlib import Path

# Main output directory for all sweep results
SWEEP_DIR = "sweep_results"

# Random seeds for robustness (only first seed gets full visualizations)
SEEDS = [7, 42, 123, 456, 789]

# Baseline configuration from README Quick start
BASE_ARGS = {
    'N': 60,
    'T': 150,
    'D': 3,
    'p_edge': 0.10,
    'p_seed': 0.02,
    'rho': 0.4,
    'drop_dim': 1,
    'c0': -4.0,
    'c1': 2.0,
    'a0': '-3.0,-3.3,-2.7',  # Self hazard baseline per dimension
    'K_self': 3,
    'K_edge': 3,
    'mode_self': 'linear',
    'mode_edge': 'linear',
    'bias_self': 10,
    'bias_edge': 10,
    'time_mode': 'warmup',  # Default to warmup mode
    'em_iters': 24,
    'epochs_self': 12,
    'epochs_edge': 10,
    'lr_self': 3e-3,
    'lr_edge': 8e-3,
    'wd_self': 1e-4,
    'wd_edge': 3e-3,
    'self_hidden': 64,
    'self_depth': 2,
    'edge_emb': 8,
    'edge_hidden': 96,
    'edge_depth': 3,
    'phi_monotone': 1,
    'tau_resp': 0.85,
    'edge_resp_floor': 0.12,
    'lambda_rank': 1.0,
    'rank_pairs': 1024,
    'rank_margin': 0.02,
    'self_weighting': 'sqrt_invfreq',
    'make_plots': 1,
    'device': 'cpu'
}


def run_experiment(overrides, outdir, seed, save_full_viz=True):
    """
    Run single experiment with specified parameter overrides.
    
    Args:
        overrides: dict of parameter overrides
        outdir: output directory
        seed: random seed
        save_full_viz: if False, skip most visualizations to save disk space
    """
    print(f"Starting experiment: {outdir} (seed={seed})")
    
    args = {**BASE_ARGS, **overrides, 'outdir': outdir, 'seed': seed}
    
    # Disable most plots for non-first seeds
    if not save_full_viz:
        args['make_plots'] = 0
    
    # Build command: use '=' for args starting with '-' to avoid parsing issues
    cmd = ['python', 'viz_eval_nn_ic.py']
    for key, val in args.items():
        # Handle boolean flags (store_true arguments)
        if isinstance(val, bool):
            if val:  # Only add flag if True
                cmd.append(f'--{key}')
            # If False, omit the argument entirely
            continue
        
        val_str = str(val)
        # Use '=' syntax for values starting with '-' or containing commas
        if val_str.startswith('-') or ',' in val_str:
            cmd.append(f'--{key}={val_str}')
        else:
            cmd.extend([f'--{key}', val_str])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {outdir}: {e}", file=sys.stderr)
        raise


def main():
    # Create main sweep directory
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f"All results will be saved to: {SWEEP_DIR}/")
    print(f"Running each config with {len(SEEDS)} seeds: {SEEDS}")
    
    # Shared density and speed configs
    density_values = [0.05, 0.10, 0.15, 0.20, 0.30]
    speed_configs_warmup = [
        {'c0': -7.0, 'c1': 0.8, 'a0': '-5.0,-5.5,-4.8', 'name': 'very_slow'},
        {'c0': -6.0, 'c1': 1.2, 'a0': '-4.2,-4.5,-4.0', 'name': 'slow'},
        {'c0': -4.0, 'c1': 2.0, 'a0': '-3.0,-3.3,-2.7', 'name': 'medium'},
        {'c0': -3.0, 'c1': 2.5, 'a0': '-2.5,-2.8,-2.2', 'name': 'fast'},
    ]
    
    # ========== WARMUP MODE EXPERIMENTS ==========
    # (A) Network density sweep: varying p_edge (warmup mode)
    print("\n" + "="*60)
    print("EXPERIMENT SET A: Network Density (p_edge) - WARMUP MODE")
    print("="*60)
    
    for p_edge in density_values:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={'p_edge': p_edge},
                outdir=os.path.join(SWEEP_DIR, 'warmup', f'density_{p_edge:.2f}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)  # Only first seed gets full viz
            )
    
    # (B) Diffusion speed sweep: varying c0, c1, and a0 (warmup mode)
    print("\n" + "="*60)
    print("EXPERIMENT SET B: Diffusion Speed (c0, c1, a0) - WARMUP MODE")
    print("="*60)
    
    for cfg in speed_configs_warmup:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={'c0': cfg['c0'], 'c1': cfg['c1'], 'a0': cfg['a0']},
                outdir=os.path.join(SWEEP_DIR, 'warmup', f'speed_{cfg["name"]}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    # ========== INVERSE-U MODE (NO TIME FEATURE) EXPERIMENTS ==========
    # (C) Inverse-U mode: density sweep (WITHOUT time feature)
    print("\n" + "="*60)
    print("EXPERIMENT SET C: Inverse-U Time Mode (Density) - NO TIME FEATURE")
    print("="*60)
    
    # Updated inverse-U config with new design
    inverseu_overrides = {
        'time_mode': 'inverse_u',
        'peak_times': '20,25,30',  # Earlier peaks for early activation
        'widths': '40,40,40',      # Medium width: start=[0,5,10], moderate speed
        'baseline': 0.05,
        'max_suppression': 5.0,    # NEW: suppression strength
        'K_self': 0,               # Disable warmup bias
        'K_edge': 0,
    }
    
    for p_edge in density_values:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**inverseu_overrides, 'p_edge': p_edge},
                outdir=os.path.join(SWEEP_DIR, 'inverse_u', f'density_{p_edge:.2f}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    # (D) Inverse-U mode: speed sweep (WITHOUT time feature)
    # Use same peak/width strategy as inverse_u_with_time for fair comparison
    print("\n" + "="*60)
    print("EXPERIMENT SET D: Inverse-U Time Mode (Speed) - NO TIME FEATURE")
    print("="*60)
    
    speed_configs_inverseu = [
        {'peak_times': '50,55,60', 'widths': '100,100,100', 'name': 'very_slow'},
        {'peak_times': '35,40,45', 'widths': '70,70,70', 'name': 'slow'},
        {'peak_times': '25,30,35', 'widths': '50,50,50', 'name': 'medium'},
        {'peak_times': '15,20,25', 'widths': '30,30,30', 'name': 'fast'},
    ]
    
    for cfg in speed_configs_inverseu:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**inverseu_overrides, 
                          'peak_times': cfg['peak_times'], 
                          'widths': cfg['widths']},
                outdir=os.path.join(SWEEP_DIR, 'inverse_u', f'speed_{cfg["name"]}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    # ========== INVERSE-U + TIME FEATURE EXPERIMENTS ==========
    print("\n" + "="*60)
    print("EXPERIMENT SET E: Inverse-U + TIME FEATURE (Density)")
    print("="*60)
    
    inverseu_time_overrides = {
        'time_mode': 'inverse_u',
        'peak_times': '20,25,30',  # Earlier peaks for early activation
        'widths': '40,40,40',      # Medium width: start=[0,5,10], moderate speed
        'baseline': 0.05,
        'max_suppression': 5.0,    # NEW: suppression strength
        'K_self': 0,               # Disable warmup bias
        'K_edge': 0,
        'use_time': True,          # ★ KEY DIFFERENCE: include time feature
    }
    
    for p_edge in density_values:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**inverseu_time_overrides, 'p_edge': p_edge},
                outdir=os.path.join(SWEEP_DIR, 'inverse_u_with_time', f'density_{p_edge:.2f}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    print("\n" + "="*60)
    print("EXPERIMENT SET F: Inverse-U + TIME FEATURE (Speed)")
    print("="*60)
    
    for cfg in speed_configs_inverseu:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**inverseu_time_overrides, 
                          'peak_times': cfg['peak_times'], 
                          'widths': cfg['widths']},
                outdir=os.path.join(SWEEP_DIR, 'inverse_u_with_time', f'speed_{cfg["name"]}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print(f"Results saved in: {SWEEP_DIR}/")
    print(f"\nSummary:")
    print(f"  - warmup/: {len(density_values) + len(speed_configs_warmup)} configs × {len(SEEDS)} seeds")
    print(f"  - inverse_u/: {len(density_values) + len(speed_configs_inverseu)} configs × {len(SEEDS)} seeds")
    print(f"  - inverse_u_with_time/: {len(density_values) + len(speed_configs_inverseu)} configs × {len(SEEDS)} seeds")
    total = 3 * (len(density_values) + len(speed_configs_inverseu)) * len(SEEDS)
    print(f"  - Total: {total} experiments")
    print("="*60)


if __name__ == '__main__':
    main()
