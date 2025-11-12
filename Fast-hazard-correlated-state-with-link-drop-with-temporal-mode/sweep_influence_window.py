#!/usr/bin/env python3
# sweep_influence_window.py
# Test influence window parameter m across three time modulation modes:
# - warmup mode
# - inverse_u mode
# - inverse_u_with_time mode
# Phase 1: Matched m (data generation m = training m)
# Phase 2: Mismatched m (data generation m ≠ training m)

import subprocess
import sys
import os
from pathlib import Path

# Main output directory for influence window experiments
SWEEP_DIR = "sweep_results_m"

# Random seeds for robustness
SEEDS = [7, 42, 123]  # Using 3 seeds for faster experiments

# Baseline configuration (warmup mode)
BASE_ARGS_WARMUP = {
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
    'K_self': 3,
    'K_edge': 3,
    'mode_self': 'linear',
    'mode_edge': 'linear',
    'bias_self': 10,
    'bias_edge': 10,
    'time_mode': 'warmup',
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

# Inverse-U mode (without time feature)
# Updated to match new inverse-U design with earlier activation
BASE_ARGS_INVERSEU = {
    **BASE_ARGS_WARMUP,
    'time_mode': 'inverse_u',
    'peak_times': '20,25,30',  # Earlier peaks for early activation
    'widths': '40,40,40',      # Medium width: start=[0,5,10], moderate diffusion speed
    'baseline': 0.05,
    'max_suppression': 5.0,    # NEW: suppression strength (logit units)
    'K_self': 0,               # Disabled - inverse-U handles early suppression
    'K_edge': 0,
}

# Inverse-U mode (with time feature)
BASE_ARGS_INVERSEU_TIME = {
    **BASE_ARGS_INVERSEU,
    'use_time': True,
}


def run_experiment(base_args, overrides, outdir, seed, save_full_viz=True):
    """Run single experiment with specified parameter overrides."""
    print(f"\n{'='*70}")
    print(f"Starting: {outdir} (seed={seed})")
    print(f"{'='*70}")
    
    args = {**base_args, **overrides, 'outdir': outdir, 'seed': seed}
    
    # Disable plots for non-first seeds
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
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed: {outdir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {outdir}: {e}", file=sys.stderr)
        raise


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f"All results will be saved to: {SWEEP_DIR}/")
    print(f"Running each config with {len(SEEDS)} seeds: {SEEDS}\n")
    
    # Test different m values
    m_values = [1, 2, 3, 5]
    
    # Define time modulation modes
    time_modes = {
        'warmup': BASE_ARGS_WARMUP,
        'inverse_u': BASE_ARGS_INVERSEU,
        'inverse_u_with_time': BASE_ARGS_INVERSEU_TIME,
    }
    
    # ========== PHASE 1: Matched m across time modes ==========
    print("\n" + "="*70)
    print("PHASE 1: MATCHED m (data generation m = training m)")
    print("Testing across three time modulation modes")
    print("="*70)
    
    for mode_name, base_args in time_modes.items():
        print(f"\n{'='*70}")
        print(f"TIME MODE: {mode_name.upper()}")
        print(f"{'='*70}")
        
        for m in m_values:
            print(f"\n--- Testing m={m} (matched) with {mode_name} ---")
            for idx, seed in enumerate(SEEDS):
                run_experiment(
                    base_args=base_args,
                    overrides={'m': m},
                    outdir=os.path.join(SWEEP_DIR, 'matched', mode_name, f'm{m}_seed{seed}'),
                    seed=seed,
                    save_full_viz=(idx == 0)
                )
    
    # ========== PHASE 2: Mismatched m (across time modes) ==========
    print("\n" + "="*70)
    print("PHASE 2: MISMATCHED m (data generation m ≠ training m)")
    print("Testing model robustness across time modulation modes")
    print("="*70)
    
    # Three mismatch scenarios
    mismatch_scenarios = [
        (3, 1, "UNDERESTIMATE"),   # True m=3, Train m=1
        (1, 3, "OVERESTIMATE"),     # True m=1, Train m=3
        (5, 2, "MODERATE UNDER"),   # True m=5, Train m=2
    ]
    
    for mode_name, base_args in time_modes.items():
        print(f"\n{'='*70}")
        print(f"Mismatch with TIME MODE: {mode_name.upper()}")
        print(f"{'='*70}")
        
        for true_m, train_m, scenario_type in mismatch_scenarios:
            print(f"\n--- Scenario: True m={true_m}, Train m={train_m} ({scenario_type}) ---")
            
            for idx, seed in enumerate(SEEDS):
                run_experiment(
                    base_args=base_args,
                    overrides={'m': true_m, 'm_train': train_m},
                    outdir=os.path.join(SWEEP_DIR, 'mismatch', mode_name, 
                                       f'true_m{true_m}_train_m{train_m}_seed{seed}'),
                    seed=seed,
                    save_full_viz=(idx == 0)
                )
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved in: {SWEEP_DIR}/")
    
    print(f"\nPHASE 1 (Matched m across time modes):")
    print(f"  - Location: {SWEEP_DIR}/matched/{{mode}}/")
    print(f"  - Time modes: warmup, inverse_u, inverse_u_with_time")
    print(f"  - m values: {m_values}")
    print(f"  - {len(time_modes)} modes × {len(m_values)} m's × {len(SEEDS)} seeds = {len(time_modes) * len(m_values) * len(SEEDS)} experiments")
    
    print(f"\nPHASE 2 (Mismatched m across time modes):")
    print(f"  - Location: {SWEEP_DIR}/mismatch/{{mode}}/")
    print(f"  - Time modes: warmup, inverse_u, inverse_u_with_time")
    print(f"  - Scenarios:")
    print(f"    • True m=3, Train m=1 (underestimate)")
    print(f"    • True m=1, Train m=3 (overestimate)")
    print(f"    • True m=5, Train m=2 (moderate underestimate)")
    print(f"  - {len(time_modes)} modes × 3 scenarios × {len(SEEDS)} seeds = {len(time_modes) * 3 * len(SEEDS)} experiments")
    
    print(f"\nTotal: {len(time_modes) * (len(m_values) + 3) * len(SEEDS)} experiments")
    print("="*70)


if __name__ == '__main__':
    main()
