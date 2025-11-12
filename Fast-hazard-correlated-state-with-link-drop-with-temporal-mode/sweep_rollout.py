#!/usr/bin/env python3
# sweep_rollout.py
# Sweep experiments for train/test split with rollout prediction
# Tests diffusion speed under three time modes: warmup, inverse_u, inverse_u_with_time

import subprocess
import sys
import os
from pathlib import Path

# Main output directory for rollout sweep results
SWEEP_DIR = "sweep_results_rollout"

# Random seeds for robustness
SEEDS = [7, 42, 123, 456, 789]

# Baseline configuration
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
    'a0': '-3.0,-3.3,-2.7',
    # Train/test split parameters
    'T_train': 25,
    'T_test': 60,
    # Training parameters
    'em_iters': 24,
    'epochs_self': 10,
    'epochs_edge': 8,
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
    """Run single rollout experiment with specified parameter overrides."""
    print(f"Starting rollout experiment: {outdir} (seed={seed})")
    
    args = {**BASE_ARGS, **overrides, 'outdir': outdir, 'seed': seed}
    
    # Disable most plots for non-first seeds
    if not save_full_viz:
        args['make_plots'] = 0
    
    # Build command
    cmd = ['python', 'viz_eval_nn_ic.py']
    for key, val in args.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f'--{key}')
            continue
        
        val_str = str(val)
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
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f"All rollout results will be saved to: {SWEEP_DIR}/")
    print(f"Running each config with {len(SEEDS)} seeds: {SEEDS}")
    
    # Diffusion speed configurations for warmup mode
    speed_configs_warmup = [
        {'c0': -7.0, 'c1': 0.8, 'a0': '-5.0,-5.5,-4.8', 'name': 'very_slow'},
        {'c0': -6.0, 'c1': 1.2, 'a0': '-4.2,-4.5,-4.0', 'name': 'slow'},
        {'c0': -4.0, 'c1': 2.0, 'a0': '-3.0,-3.3,-2.7', 'name': 'medium'},
        {'c0': -3.0, 'c1': 2.5, 'a0': '-2.5,-2.8,-2.2', 'name': 'fast'},
    ]
    
    # Speed configs for inverse-U mode (control via peak_times & widths)
    # Adjusted for T_train=20, T_test=50: ensure significant activity during training
    speed_configs_inverseu = [
        {'peak_times': '30,35,40', 'widths': '60,60,60', 'name': 'very_slow'},
        {'peak_times': '22,27,32', 'widths': '44,44,44', 'name': 'slow'},
        {'peak_times': '16,21,26', 'widths': '32,32,32', 'name': 'medium'},
        {'peak_times': '12,17,22', 'widths': '24,24,24', 'name': 'fast'},
    ]
    
    # ===== MODE 1: Warmup =====
    print("\n" + "="*60)
    print("ROLLOUT EXPERIMENTS: WARMUP MODE")
    print("="*60)
    
    warmup_overrides = {
        'time_mode': 'warmup',
        'K_self': 3,
        'K_edge': 3,
        'mode_self': 'linear',
        'mode_edge': 'linear',
        'bias_self': 10,
        'bias_edge': 10,
    }
    
    for cfg in speed_configs_warmup:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**warmup_overrides, 'c0': cfg['c0'], 'c1': cfg['c1'], 'a0': cfg['a0']},
                outdir=os.path.join(SWEEP_DIR, 'warmup', f'speed_{cfg["name"]}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    # ===== MODE 2: Inverse-U (no time feature) =====
    print("\n" + "="*60)
    print("ROLLOUT EXPERIMENTS: INVERSE-U MODE (no time feature)")
    print("="*60)
    
    inverseu_overrides = {
        'time_mode': 'inverse_u',
        'peak_times': '20,25,30',  # Default (will be overridden)
        'widths': '40,40,40',      # Default (will be overridden)
        'baseline': 0.05,
        'max_suppression': 5.0,    # NEW: suppression strength
        'K_self': 0,               # Disabled - inverse-U handles early suppression
        'K_edge': 0,
    }
    
    for cfg in speed_configs_inverseu:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**inverseu_overrides, 'peak_times': cfg['peak_times'], 'widths': cfg['widths']},
                outdir=os.path.join(SWEEP_DIR, 'inverse_u', f'speed_{cfg["name"]}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    # ===== MODE 3: Inverse-U + Time Feature =====
    print("\n" + "="*60)
    print("ROLLOUT EXPERIMENTS: INVERSE-U + TIME FEATURE MODE")
    print("="*60)
    
    inverseu_time_overrides = {
        'time_mode': 'inverse_u',
        'peak_times': '20,25,30',  # Default (will be overridden)
        'widths': '40,40,40',      # Default (will be overridden)
        'baseline': 0.05,
        'max_suppression': 5.0,    # NEW: suppression strength
        'K_self': 0,               # Disabled - inverse-U handles early suppression
        'K_edge': 0,
        'use_time': True,          # ★ KEY DIFFERENCE
    }
    
    for cfg in speed_configs_inverseu:
        for idx, seed in enumerate(SEEDS):
            run_experiment(
                overrides={**inverseu_time_overrides, 'peak_times': cfg['peak_times'], 'widths': cfg['widths']},
                outdir=os.path.join(SWEEP_DIR, 'inverse_u_with_time', f'speed_{cfg["name"]}_seed{seed}'),
                seed=seed,
                save_full_viz=(idx == 0)
            )
    
    print("\n" + "="*60)
    print("All rollout experiments completed successfully!")
    print(f"Results saved in: {SWEEP_DIR}/")
    print(f"  - warmup/ : {len(speed_configs_warmup)} configs × {len(SEEDS)} seeds = {len(speed_configs_warmup) * len(SEEDS)} experiments")
    print(f"  - inverse_u/ : {len(speed_configs_inverseu)} configs × {len(SEEDS)} seeds = {len(speed_configs_inverseu) * len(SEEDS)} experiments")
    print(f"  - inverse_u_with_time/ : {len(speed_configs_inverseu)} configs × {len(SEEDS)} seeds = {len(speed_configs_inverseu) * len(SEEDS)} experiments")
    print(f"Total: {(len(speed_configs_warmup) + 2 * len(speed_configs_inverseu)) * len(SEEDS)} experiments")
    print("="*60)


if __name__ == '__main__':
    main()
