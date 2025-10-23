# generator_density_seed_sweep.py - COMPLETE VERSION
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from main_configurable import run_scenario
from config import ALL_GEN_DENSITY_SEED_SWEEP, GEN_DENSITIES, GEN_SEED_RATIOS

def run_generator_k_times_get_trajectories(config, k_runs=10):
    """
    Run generator k times and return averaged trajectories
    
    Returns:
        dict with trajectory data and final percentages
    """
    all_trajectories = []
    repair_final_pcts = []
    vacant_final_pcts = []
    sell_final_pcts = []
    actual_densities = []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(os.path.dirname(current_dir), 'generator')
    
    for run_id in range(k_runs):
        # Create modified config with unique seed
        from config import ScenarioConfig
        run_config = ScenarioConfig(
            name=f"{config.name}_temp_run{run_id}",
            description=config.description,
            alpha_bonding=config.alpha_bonding,
            beta_bridging=config.beta_bridging,
            gamma_decay=config.gamma_decay,
            target_seed_ratio=config.target_seed_ratio,
            jitter_fraction=config.jitter_fraction,
            repair_base_prob=config.repair_base_prob,
            repair_damage_coeff=config.repair_damage_coeff,
            repair_building_coeff=config.repair_building_coeff,
            repair_income_coeff=config.repair_income_coeff,
            vacant_base_prob=config.vacant_base_prob,
            vacant_damage_coeff=config.vacant_damage_coeff,
            vacant_income_coeff=config.vacant_income_coeff,
            vacant_age_coeff=config.vacant_age_coeff,
            sales_base_prob=config.sales_base_prob,
            sales_damage_coeff=config.sales_damage_coeff,
            sales_building_coeff=config.sales_building_coeff,
            sales_age_coeff=config.sales_age_coeff,
            n_households=config.n_households,
            time_horizon=config.time_horizon,
            p_block=config.p_block,
            blocking_mode=config.blocking_mode,
            L=config.L,
            target_avg_degree=config.target_avg_degree,
            random_seed=config.random_seed + run_id * 1000
        )
        
        # Run scenario and get stats
        stats = run_scenario(run_config, base_data_path)
        
        # Load generated states to get trajectory
        temp_folder = os.path.join(os.path.dirname(current_dir), 'dataset', run_config.name)
        states_file = os.path.join(temp_folder, 'household_states_raw.csv')
        states = pd.read_csv(states_file)
        
        total_households = len(states[states['time'] == 0])
        
        # Collect trajectory
        trajectory = []
        for t in range(config.time_horizon):
            states_t = states[states['time'] == t]
            trajectory.append({
                'timestep': t,
                'repair_pct': (states_t['repair_state'].sum() / total_households) * 100,
                'vacant_pct': (states_t['vacancy_state'].sum() / total_households) * 100,
                'sell_pct': (states_t['sales_state'].sum() / total_households) * 100
            })
        all_trajectories.append(pd.DataFrame(trajectory))
        
        # Final percentages
        repair_final_pcts.append(stats['repair_pct_tT'])
        vacant_final_pcts.append(stats['vacant_pct_tT'])
        sell_final_pcts.append(stats['sell_pct_tT'])
        actual_densities.append(stats['avg_degree_t0'])
        
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
    
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

def generate_generator_comparison_plots(k_runs=10, output_dir='./generator_density_seed_sweep'):
    """
    Generate four plots for generator:
    - 3 time evolution plots (one per decision type)
    - 1 final percentages plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generator Density×Seed Sweep Analysis")
    print(f"{'='*70}")
    print(f"K runs per scenario: {k_runs}")
    print(f"Total scenarios: {len(ALL_GEN_DENSITY_SEED_SWEEP)}")
    print(f"Total simulations: {len(ALL_GEN_DENSITY_SEED_SWEEP) * k_runs}\n")
    
    # Collect all data
    results = []
    for scenario in tqdm(ALL_GEN_DENSITY_SEED_SWEEP, desc="Running scenarios"):
        stats = run_generator_k_times_get_trajectories(scenario, k_runs=k_runs)
        results.append(stats)
    
    df_results = pd.DataFrame([{
        'density': r['density_mean'],
        'seed_ratio': r['seed_ratio'],
        'trajectory_mean': r['trajectory_mean'],
        'trajectory_std': r['trajectory_std'],
        'repair_final_mean': r['repair_final_mean'],
        'repair_final_std': r['repair_final_std'],
        'vacant_final_mean': r['vacant_final_mean'],
        'vacant_final_std': r['vacant_final_std'],
        'sell_final_mean': r['sell_final_mean'],
        'sell_final_std': r['sell_final_std']
    } for r in results])
    
    # =================================================================
    # PLOTS 1-3: Time Evolution (one plot per decision type)
    # One subplot per seed ratio
    # =================================================================
    for decision in ['repair', 'vacant', 'sell']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for col_idx, seed_ratio in enumerate([0.05, 0.20]):
            ax = axes[col_idx]
            df_seed = df_results[df_results['seed_ratio'] == seed_ratio]
            
            # Plot each density as a separate line
            for _, row in df_seed.iterrows():
                density = row['density']
                traj_mean = row['trajectory_mean']
                timesteps = traj_mean['timestep']
                values = traj_mean[f'{decision}_pct']
                
                ax.plot(timesteps, values, 
                       label=f'D={density:.1f}', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{decision.capitalize()} Percentage (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'Seed Ratio: {seed_ratio:.0%}', fontsize=12, fontweight='bold')
            ax.legend(title='Density', fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.suptitle(f'Generator: {decision.capitalize()} Decision Evolution Over Time (Avg of {k_runs} runs)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'generator_time_evolution_{decision}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{decision.capitalize()} time evolution saved: {output_path}")
    
    # =================================================================
    # PLOT 4: Final Percentages (1×2 grid)
    # =================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'repair': '#1f77b4', 'vacant': '#ff7f0e', 'sell': '#2ca02c'}
    markers = {'repair': 'o', 'vacant': 's', 'sell': '^'}
    
    for col_idx, seed_ratio in enumerate([0.05, 0.20]):
        ax = axes2[col_idx]
        df_seed = df_results[df_results['seed_ratio'] == seed_ratio].sort_values('density')
        
        for decision in ['repair', 'vacant', 'sell']:
            ax.plot(df_seed['density'], df_seed[f'{decision}_final_mean'],
                   marker=markers[decision], label=decision.capitalize(),
                   linewidth=2.5, markersize=8, 
                   color=colors[decision], alpha=0.8)
            
            ax.fill_between(df_seed['density'],
                          df_seed[f'{decision}_final_mean'] - df_seed[f'{decision}_final_std'],
                          df_seed[f'{decision}_final_mean'] + df_seed[f'{decision}_final_std'],
                          alpha=0.15, color=colors[decision])
        
        ax.set_xlabel('Network Density (Avg Degree)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Final Decision Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Seed Ratio: {seed_ratio:.0%}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.suptitle(f'Generator: Final Decisions vs Network Density (Avg of {k_runs} runs)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'generator_final_percentages.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()


    # =================================================================
    # NEW PLOTS 5-7: New Adopters Per Timestep (incremental)
    # =================================================================
    for decision in ['repair', 'vacant', 'sell']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for col_idx, seed_ratio in enumerate([0.05, 0.20]):
            ax = axes[col_idx]
            df_seed = df_results[df_results['seed_ratio'] == seed_ratio]
            
            # Plot each density as a separate line
            for _, row in df_seed.iterrows():
                density = row['density']
                traj_mean = row['trajectory_mean']
                timesteps = traj_mean['timestep'].values
                
                # Calculate new adopters (difference between consecutive timesteps)
                cumulative = traj_mean[f'{decision}_pct'].values
                new_adopters = np.concatenate([[cumulative[0]], np.diff(cumulative)])
                
                ax.plot(timesteps, new_adopters, 
                       label=f'D={density:.1f}', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'New {decision.capitalize()} Adopters (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'Seed Ratio: {seed_ratio:.0%}', fontsize=12, fontweight='bold')
            ax.legend(title='Density', fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        plt.suptitle(f'Generator: New {decision.capitalize()} Adopters Per Timestep (Avg of {k_runs} runs)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'generator_new_adopters_{decision}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nGenerator {decision.capitalize()} new adopters saved: {output_path}")
    
    print(f"Final percentages saved: {output_path2}")
    print(f"\n{'='*70}")
    print("Generator analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Run ABM sweep: 30 scenarios × 10 runs = 300 simulations
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_generator_comparison_plots(k_runs=10, output_dir=current_dir+'/density_seed_sweep_results')

