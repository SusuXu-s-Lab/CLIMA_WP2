"""
multi_run_analysis.py - Generate multiple datasets per scenario and create averaged visualizations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from config import ScenarioConfig, ALL_SCENARIOS
from main_configurable import run_scenario

def generate_multiple_runs(config: ScenarioConfig, base_data_path: str, num_runs: int = 10):
    """
    Generate multiple datasets for a single scenario with different random seeds
    
    Args:
        config: ScenarioConfig object
        base_data_path: Path to base data directory
        num_runs: Number of runs to generate
        
    Returns:
        List of paths to generated datasets
    """
    original_seed = config.random_seed
    original_name = config.name
    dataset_paths = []
    
    print(f"\n{'='*60}")
    print(f"Generating {num_runs} runs for scenario: {original_name}")
    print(f"{'='*60}")
    
    for run_idx in range(num_runs):
        # Create modified config with new seed and name
        run_seed = original_seed + run_idx
        run_config = ScenarioConfig(
            name=f"{original_name}_run{run_idx}",
            description=f"{config.description} (Run {run_idx})",
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
            random_seed=run_seed
        )
        
        print(f"\n--- Run {run_idx+1}/{num_runs} (seed={run_seed}) ---")
        run_scenario(run_config, base_data_path)
        
        # Store path to generated data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(os.path.dirname(current_dir), 'dataset', run_config.name)
        dataset_paths.append(dataset_path)
    
    return dataset_paths

def calculate_new_adopters(states_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate new decision adopters at each timestep
    
    Args:
        states_df: DataFrame with household states over time
        
    Returns:
        DataFrame with new adopters counts per timestep
    """
    states_df = states_df.copy()
    states_df['time'] = states_df['time'].astype(int)
    
    new_adopters = {'time': [], 'repair': [], 'vacancy': [], 'sales': []}
    times = sorted(states_df['time'].unique())
    
    for t in times:
        if t == 0:
            df_t0 = states_df[states_df['time'] == 0]
            new_adopters['time'].append(t)
            new_adopters['repair'].append(int(df_t0['repair_state'].sum()))
            new_adopters['vacancy'].append(int(df_t0['vacancy_state'].sum()))
            new_adopters['sales'].append(int(df_t0['sales_state'].sum()))
        else:
            df_prev = states_df[states_df['time'] == t-1].set_index('home')
            df_curr = states_df[states_df['time'] == t].set_index('home')
            
            repair_new = int(((df_curr['repair_state'] == 1) & (df_prev['repair_state'] == 0)).sum())
            vacancy_new = int(((df_curr['vacancy_state'] == 1) & (df_prev['vacancy_state'] == 0)).sum())
            sales_new = int(((df_curr['sales_state'] == 1) & (df_prev['sales_state'] == 0)).sum())
            
            new_adopters['time'].append(t)
            new_adopters['repair'].append(repair_new)
            new_adopters['vacancy'].append(vacancy_new)
            new_adopters['sales'].append(sales_new)
    
    return pd.DataFrame(new_adopters)

def visualize_averaged_new_adopters(dataset_paths: list, scenario_name: str, output_path: str):
    """
    Create visualization of averaged new adopters across multiple runs
    
    Args:
        dataset_paths: List of paths to dataset folders
        scenario_name: Name of the scenario
        output_path: Path to save the plot
    """
    all_runs_data = []
    
    print(f"\nLoading data from {len(dataset_paths)} runs...")
    for path in tqdm(dataset_paths):
        states_file = os.path.join(path, 'household_states_raw.csv')
        if os.path.exists(states_file):
            states_df = pd.read_csv(states_file)
            adopters_df = calculate_new_adopters(states_df)
            all_runs_data.append(adopters_df)
    
    if not all_runs_data:
        print("No data found!")
        return
    
    # Calculate mean and std across runs
    times = all_runs_data[0]['time'].values
    repair_runs = np.array([df['repair'].values for df in all_runs_data])
    vacancy_runs = np.array([df['vacancy'].values for df in all_runs_data])
    sales_runs = np.array([df['sales'].values for df in all_runs_data])
    
    repair_mean = repair_runs.mean(axis=0)
    repair_std = repair_runs.std(axis=0)
    vacancy_mean = vacancy_runs.mean(axis=0)
    vacancy_std = vacancy_runs.std(axis=0)
    sales_mean = sales_runs.mean(axis=0)
    sales_std = sales_runs.std(axis=0)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot repair
    plt.plot(times, repair_mean, 'o-', label='Repair', linewidth=2, markersize=6, color='tab:blue')
    plt.fill_between(times, repair_mean - repair_std, repair_mean + repair_std, 
                     alpha=0.2, color='tab:blue')
    
    # Plot vacancy
    plt.plot(times, vacancy_mean, 's-', label='Vacancy', linewidth=2, markersize=6, color='tab:orange')
    plt.fill_between(times, vacancy_mean - vacancy_std, vacancy_mean + vacancy_std, 
                     alpha=0.2, color='tab:orange')
    
    # Plot sales
    plt.plot(times, sales_mean, '^-', label='Sales', linewidth=2, markersize=6, color='tab:green')
    plt.fill_between(times, sales_mean - sales_std, sales_mean + sales_std, 
                     alpha=0.2, color='tab:green')
    
    plt.title(f"Average New Decision Adopters per Timestep - {scenario_name}\n(Mean ± Std over {len(all_runs_data)} runs)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Number of New Adopters", fontsize=12)
    plt.legend(title='Decision Type', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Averaged visualization saved: {output_path}")

def run_multi_run_analysis(config: ScenarioConfig, base_data_path: str, num_runs: int = 10):
    """
    Complete workflow: generate multiple runs and create averaged visualization
    
    Args:
        config: ScenarioConfig object
        base_data_path: Path to base data directory
        num_runs: Number of runs to generate
    """
    # Generate multiple runs
    dataset_paths = generate_multiple_runs(config, base_data_path, num_runs)
    
    # Create output path for averaged visualization
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_path = os.path.join(parent_dir, 'dataset', f'{config.name}_averaged_new_adopters.png')
    
    # Create averaged visualization
    visualize_averaged_new_adopters(dataset_paths, config.name, output_path)
    
    return dataset_paths

def run_multi_run_analysis_all_scenarios(num_runs: int = 10):
    """
    Run multi-run analysis for all scenarios
    
    Args:
        num_runs: Number of runs per scenario
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(os.path.dirname(current_dir), 'generator')
    
    for config in ALL_SCENARIOS:
        print(f"\n{'#'*60}")
        print(f"Processing scenario: {config.name}")
        print(f"{'#'*60}")
        run_multi_run_analysis(config, base_data_path, num_runs)

if __name__ == "__main__":
    # Run multi-run analysis for all scenarios (10 runs each)
    run_multi_run_analysis_all_scenarios(num_runs=10)