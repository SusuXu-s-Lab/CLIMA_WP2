# density_sweep_analysis.py - Two comprehensive 3×2 plots
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from abm_data_generator_configurable import ABMDataGenerator
from abm_config import (
    DENSITY_SEED_SWEEP_HIGH_RESISTANT,
    DENSITY_SEED_SWEEP_BALANCED,
    DENSITY_SEED_SWEEP_HIGH_SOCIAL,
    DENSITIES,
    SEED_RATIOS
)

def run_scenario_k_times_get_full_trajectories(config, k_runs=10):
    """
    Run scenario k times and return averaged trajectories over time
    
    Returns:
        dict with time-series data and final percentages
    """
    all_trajectories = []
    repair_final_pcts = []
    vacant_final_pcts = []
    sell_final_pcts = []
    actual_densities = []
    
    for run_id in range(k_runs):
        generator = ABMDataGenerator(config)
        np.random.seed(config.random_seed + run_id * 1000)
        data = generator.simulate()
        
        states = data['household_states']
        total_households = len(states[states['timestep'] == 0])
        
        # Collect trajectory for this run
        trajectory = []
        for t in range(config.n_timesteps):
            states_t = states[states['timestep'] == t]
            trajectory.append({
                'timestep': t,
                'repair_count': states_t['repair'].sum(),
                'vacant_count': states_t['vacant'].sum(),
                'sell_count': states_t['sell'].sum(),
                'repair_pct': (states_t['repair'].sum() / total_households) * 100,
                'vacant_pct': (states_t['vacant'].sum() / total_households) * 100,
                'sell_pct': (states_t['sell'].sum() / total_households) * 100
            })
        all_trajectories.append(pd.DataFrame(trajectory))
        
        # Final percentages
        states_final = states[states['timestep'] == config.n_timesteps - 1]
        repair_final_pcts.append((states_final['repair'].sum() / total_households) * 100)
        vacant_final_pcts.append((states_final['vacant'].sum() / total_households) * 100)
        sell_final_pcts.append((states_final['sell'].sum() / total_households) * 100)
        
        # Actual density
        network = data['ground_truth_network']
        network_t0 = network[network['timestep'] == 0]
        actual_densities.append((2 * len(network_t0)) / total_households)
    
    # Average trajectories across runs
    avg_trajectory = pd.concat(all_trajectories).groupby('timestep').mean().reset_index()
    std_trajectory = pd.concat(all_trajectories).groupby('timestep').std().reset_index()
    
    return {
        'density_mean': np.mean(actual_densities),
        'seed_ratio': config.target_seed_ratio,
        'trajectory_mean': avg_trajectory,
        'trajectory_std': std_trajectory,
        'repair_final_mean': np.mean(repair_final_pcts),
        'repair_final_std': np.std(repair_final_pcts),
        'vacant_final_mean': np.mean(vacant_final_pcts),
        'vacant_final_std': np.std(vacant_final_pcts),
        'sell_final_mean': np.mean(sell_final_pcts),
        'sell_final_std': np.std(sell_final_pcts)
    }

def generate_comprehensive_comparison_plots(k_runs=20, output_dir='./density_seed_sweep_results'):
    """
    Generate two 3×2 comprehensive plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs = {
        'High Resistant (25%)': DENSITY_SEED_SWEEP_HIGH_RESISTANT,
        'Balanced (40%)': DENSITY_SEED_SWEEP_BALANCED,
        'High Social (70%)': DENSITY_SEED_SWEEP_HIGH_SOCIAL
    }
    
    print(f"\n{'='*70}")
    print(f"Comprehensive Density × Seed Ratio Analysis")
    print(f"{'='*70}")
    print(f"K runs per scenario: {k_runs}")
    print(f"Densities: {DENSITIES}")
    print(f"Seed Ratios: {SEED_RATIOS}")
    print(f"Total simulations: {len(DENSITY_SEED_SWEEP_HIGH_RESISTANT + DENSITY_SEED_SWEEP_BALANCED + DENSITY_SEED_SWEEP_HIGH_SOCIAL) * k_runs}\n")
    
    # Collect all data
    all_results = {}
    for config_name, scenarios in configs.items():
        print(f"\nProcessing: {config_name}")
        results = []
        for scenario in tqdm(scenarios, desc=f"  Running scenarios"):
            stats = run_scenario_k_times_get_full_trajectories(scenario, k_runs=k_runs)
            results.append(stats)
        all_results[config_name] = results
    
    # =================================================================
    # PLOTS 1-3: Time Evolution - ONE PLOT PER DECISION TYPE
    # Each plot is 3×2 grid (cleaner with only 5 lines per subplot)
    # =================================================================
    for decision in ['repair', 'vacant', 'sell']:
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        
        for row_idx, config_name in enumerate(['High Resistant (25%)', 'Balanced (40%)', 'High Social (70%)']):
            results = all_results[config_name]
            df_results = pd.DataFrame([{
                'density': r['density_mean'],
                'seed_ratio': r['seed_ratio'],
                'trajectory_mean': r['trajectory_mean'],
                'trajectory_std': r['trajectory_std']
            } for r in results])
            
            for col_idx, seed_ratio in enumerate([0.05, 0.20]):
                ax = axes[row_idx, col_idx]
                df_seed = df_results[df_results['seed_ratio'] == seed_ratio]
                
                # Plot only ONE decision type, different lines for different densities
                for _, row in df_seed.iterrows():
                    density = row['density']
                    traj_mean = row['trajectory_mean']
                    timesteps = traj_mean['timestep']
                    
                    ax.plot(timesteps, traj_mean[f'{decision}_pct'], 
                           label=f'D={density:.1f}', linewidth=2, alpha=0.8)
                
                # Formatting
                if row_idx == 0:
                    ax.set_title(f'Seed Ratio: {seed_ratio:.0%}', fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f'{config_name}\n{decision.capitalize()} (%)', 
                                 fontsize=10, fontweight='bold')
                if row_idx == 2:
                    ax.set_xlabel('Time Step', fontsize=10)
                
                ax.legend(title='Density', fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
        
        plt.suptitle(f'{decision.capitalize()} Decision Evolution Over Time (Avg of {k_runs} runs)\nComparing Network Densities Across Role Distributions and Seed Ratios', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'time_evolution_{decision}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{decision.capitalize()} time evolution plot saved: {output_path}")
    
    # =================================================================
    # PLOT 4: Final Percentages (same as before - all three decisions)
    # =================================================================
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 15))
    
    for row_idx, config_name in enumerate(['High Resistant (25%)', 'Balanced (40%)', 'High Social (70%)']):
        results = all_results[config_name]
        df_results = pd.DataFrame(results)
        
        for col_idx, seed_ratio in enumerate([0.05, 0.20]):
            ax = axes2[row_idx, col_idx]
            df_seed = df_results[df_results['seed_ratio'] == seed_ratio].sort_values('density_mean')
            
            # Plot three decision types with shaded bands
            ax.plot(df_seed['density_mean'], df_seed['repair_final_mean'], 
                   'o-', label='Repair', linewidth=2.5, markersize=8, color='#1f77b4')
            ax.fill_between(df_seed['density_mean'],
                          df_seed['repair_final_mean'] - df_seed['repair_final_std'],
                          df_seed['repair_final_mean'] + df_seed['repair_final_std'],
                          alpha=0.2, color='#1f77b4')
            print("std", df_seed['repair_final_std'])
            
            ax.plot(df_seed['density_mean'], df_seed['vacant_final_mean'], 
                   's-', label='Vacant', linewidth=2.5, markersize=8, color='#ff7f0e')
            ax.fill_between(df_seed['density_mean'],
                          df_seed['vacant_final_mean'] - df_seed['vacant_final_std'],
                          df_seed['vacant_final_mean'] + df_seed['vacant_final_std'],
                          alpha=0.2, color='#ff7f0e')
            
            ax.plot(df_seed['density_mean'], df_seed['sell_final_mean'], 
                   '^-', label='Sell', linewidth=2.5, markersize=8, color='#2ca02c')
            ax.fill_between(df_seed['density_mean'],
                          df_seed['sell_final_mean'] - df_seed['sell_final_std'],
                          df_seed['sell_final_mean'] + df_seed['sell_final_std'],
                          alpha=0.2, color='#2ca02c')
            
            # Formatting
            if row_idx == 0:
                ax.set_title(f'Seed Ratio: {seed_ratio:.0%}', fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{config_name}\nFinal Percentage (%)', fontsize=10, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel('Network Density (Avg Degree)', fontsize=10)
            
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
    
    plt.suptitle(f'ABM: Final Decision Percentages vs Network Density (Avg of {k_runs} runs)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'abm_final_percentages.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()


    # =================================================================
    # NEW PLOTS 5-7: New Adopters Per Timestep (incremental, not cumulative)
    # =================================================================
    for decision in ['repair', 'vacant', 'sell']:
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        
        for row_idx, config_name in enumerate(['High Resistant (25%)', 'Balanced (40%)', 'High Social (70%)']):
            results = all_results[config_name]
            df_results = pd.DataFrame([{
                'density': r['density_mean'],
                'seed_ratio': r['seed_ratio'],
                'trajectory_mean': r['trajectory_mean'],
                'trajectory_std': r['trajectory_std']
            } for r in results])
            
            for col_idx, seed_ratio in enumerate([0.05, 0.20]):
                ax = axes[row_idx, col_idx]
                df_seed = df_results[df_results['seed_ratio'] == seed_ratio]
                
                for _, row in df_seed.iterrows():
                    density = row['density']
                    traj_mean = row['trajectory_mean']
                    timesteps = traj_mean['timestep'].values
                    
                    # Calculate new adopters (difference between consecutive timesteps)
                    cumulative = traj_mean[f'{decision}_pct'].values
                    new_adopters = np.concatenate([[cumulative[0]], np.diff(cumulative)])
                    
                    ax.plot(timesteps, new_adopters, 
                           label=f'D={density:.1f}', linewidth=2, alpha=0.8)
                
                # Formatting
                if row_idx == 0:
                    ax.set_title(f'Seed Ratio: {seed_ratio:.0%}', fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f'{config_name}\nNew {decision.capitalize()} (%)', 
                                 fontsize=10, fontweight='bold')
                if row_idx == 2:
                    ax.set_xlabel('Time Step', fontsize=10)
                
                ax.legend(title='Density', fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)  # Start from 0, auto-scale top
        
        plt.suptitle(f'ABM: New {decision.capitalize()} Adopters Per Timestep (Avg of {k_runs} runs)\nComparing Network Densities Across Role Distributions and Seed Ratios', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'abm_new_adopters_{decision}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nABM {decision.capitalize()} new adopters plot saved: {output_path}")
    
    print(f"Final percentages saved: {output_path2}")
    print(f"\n{'='*70}")
    print("ABM analysis complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Run analysis: 30 scenarios × 10 runs = 300 simulations
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_comprehensive_comparison_plots(k_runs=10, output_dir=current_dir+'/density_seed_sweep_results_2')