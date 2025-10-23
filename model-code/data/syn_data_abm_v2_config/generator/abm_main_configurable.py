# abm_main_configurable.py
import os
import numpy as np
import pandas as pd
import copy
import pickle
# from abm_data_generator_improved_2 import ABMDataGenerator, save_abm_data
from abm_data_generator_configurable import ABMDataGenerator, save_abm_data
from abm_config import ABMScenarioConfig, ALL_ABM_SCENARIOS, get_abm_scenario
from abm_dataset_statistics import (
    calculate_abm_statistics, 
    print_abm_statistics, 
    save_abm_statistics,
    create_abm_summary_excel
)

def run_abm_scenario(config: ABMScenarioConfig, base_output_path: str):
    """
    Run ABM simulation for a single scenario configuration

    Args:
        config: ABMScenarioConfig object with all parameters
        base_output_path: Base path for output (e.g., './dataset')
    
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*70}")
    print(f"Running ABM scenario: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*70}")

    # Set random seed
    np.random.seed(config.random_seed)

    # Create output directory for this scenario
    scenario_output_dir = os.path.join(base_output_path, config.name)
    os.makedirs(scenario_output_dir, exist_ok=True)
    print(f"Output directory: {scenario_output_dir}")

    # Initialize generator
    generator = ABMDataGenerator(config)

    # Run simulation
    data = generator.simulate()

    # Save data
    save_abm_data(data, scenario_output_dir, config)

    # Calculate statistics
    stats = calculate_abm_statistics(
        data['household_states'],
        data['ground_truth_network'],
        config,
        missing_stats=data.get('missing_statistics', None)
    )

    # Save statistics
    save_abm_statistics(stats, scenario_output_dir)

    # Print statistics
    print_abm_statistics(stats)

    print_decision_combination_statistics(
        data['household_states'],
        timestep=None,  # Use final timestep
        scenario_name=config.name
    )

    print(f"\nScenario {config.name} completed successfully!")
    print(f"Results saved to: {scenario_output_dir}")

    return stats

def run_abm_scenario_multiseed(config: ABMScenarioConfig, 
                               base_output_path: str,
                               n_runs: int = 10,
                               save_full_data: bool = False):
    """
    Run ABM simulation multiple times with different random seeds

    Args:
        config: ABMScenarioConfig object with all parameters
        base_output_path: Base path for output (e.g., './dataset')
        n_runs: Number of runs with different seeds (default: 10)
        save_full_data: If True, save complete datasets for each run.
                       If False (default), only save states for plotting.
    
    Returns:
        Dictionary with aggregated statistics
    """
    print(f"\n{'='*70}")
    print(f"Running ABM Multi-Seed Scenario: {config.name}")
    print(f"Description: {config.description}")
    print(f"Number of runs: {n_runs}")
    print(f"Save full data: {save_full_data}")
    print(f"{'='*70}")

    # Create scenario directory
    scenario_output_dir = os.path.join(base_output_path, config.name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    # Storage for multi-seed data
    all_stats = []
    states_cache = []  # Store household_states from all runs

    # Run simulation n_runs times with different seeds
    for i in range(n_runs):
        print(f"\n{'#'*70}")
        print(f"# Run {i+1}/{n_runs} - Seed: {config.random_seed + i*1000}")
        print(f"{'#'*70}")
    
        # Create modified config with new seed
        modified_config = copy.deepcopy(config)
        modified_config.random_seed = config.random_seed + i * 1000
    
        # Set numpy seed
        np.random.seed(modified_config.random_seed)
    
        # Initialize and run generator
        generator = ABMDataGenerator(modified_config)
        data = generator.simulate()
    
        # Always store household_states for plotting
        states_cache.append(data['household_states'].copy())
    
        # Calculate statistics
        stats = calculate_abm_statistics(
            data['household_states'],
            data['ground_truth_network'],
            modified_config,
            missing_stats=data.get('missing_statistics', None)
        )
        stats['run_id'] = i
        stats['run_seed'] = modified_config.random_seed
        all_stats.append(stats)
    
        # Optionally save full data
        if save_full_data:
            run_output_dir = os.path.join(scenario_output_dir, f'run_{i}')
            os.makedirs(run_output_dir, exist_ok=True)
            save_abm_data(data, run_output_dir, modified_config)
            print(f"  Full data saved to: {run_output_dir}")
        else:
            print(f"  Run {i+1} completed (data kept in memory only)")

    # Save states cache (always)
    cache_path = os.path.join(scenario_output_dir, 'multiseed_states_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(states_cache, f)
    print(f"\nStates cache saved: {cache_path}")

    # Aggregate statistics
    print(f"\n{'='*70}")
    print("AGGREGATING STATISTICS ACROSS RUNS")
    print(f"{'='*70}")

    aggregated_stats = {
        'scenario_name': config.name,
        'description': config.description,
        'n_runs': n_runs,
        'base_seed': config.random_seed,
        'individual_runs': all_stats,
        'aggregated_metrics': {}
    }

    # Calculate mean and std for key metrics
    metrics_to_aggregate = [
        'total_seeds_t0', 'repair_t0', 'vacant_t0', 'sell_t0',
        'repair_tT', 'vacant_tT', 'sell_tT',
        'total_links_t0', 'bonding_links_t0', 'bridging_links_t0',
        'total_links_tT', 'bonding_links_tT', 'bridging_links_tT',
        'avg_degree_t0', 'avg_degree_tT'
    ]

    for metric in metrics_to_aggregate:
        values = [s[metric] for s in all_stats]
        aggregated_stats['aggregated_metrics'][f'{metric}_mean'] = np.mean(values)
        aggregated_stats['aggregated_metrics'][f'{metric}_std'] = np.std(values)

    # Save aggregated statistics
    summary_path = os.path.join(scenario_output_dir, 'multiseed_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    print(f"Aggregated statistics saved: {summary_path}")

    # Print summary
    print(f"\nMulti-Seed Summary for {config.name}:")
    print(f"  Runs completed: {n_runs}")
    print(f"  Seeds used: {config.random_seed} to {config.random_seed + (n_runs-1)*1000}")
    print(f"\nKey Metrics (mean ± std):")
    agg = aggregated_stats['aggregated_metrics']
    print(f"  Total seeds at t=0: {agg['total_seeds_t0_mean']:.1f} ± {agg['total_seeds_t0_std']:.1f}")
    print(f"  Total links at t=0: {agg['total_links_t0_mean']:.1f} ± {agg['total_links_t0_std']:.1f}")
    print(f"  Avg degree at t=0: {agg['avg_degree_t0_mean']:.2f} ± {agg['avg_degree_t0_std']:.2f}")
    print(f"  Total links at t=T: {agg['total_links_tT_mean']:.1f} ± {agg['total_links_tT_std']:.1f}")

    print(f"\nScenario {config.name} multi-seed runs completed successfully!")

    return aggregated_stats

def print_decision_combination_statistics(household_states: pd.DataFrame, 
                                         timestep: int = None,
                                         scenario_name: str = ""):
    """
    Print detailed statistics of decision combinations at a specific timestep
    
    Args:
        household_states: DataFrame with columns [household_id, timestep, repair, vacant, sell]
        timestep: Specific timestep to analyze (if None, uses final timestep)
        scenario_name: Name of the scenario for display
    """
    if timestep is None:
        timestep = household_states['timestep'].max()
    
    # Filter to specific timestep
    states_t = household_states[household_states['timestep'] == timestep].copy()
    total_households = len(states_t)
    
    # Calculate total decisions per household
    states_t['total_decisions'] = (
        states_t['repair'] + states_t['vacant'] + states_t['sell']
    )
    
    print(f"\n{'='*70}")
    print(f"Decision Combination Statistics - {scenario_name}")
    print(f"Timestep: t={timestep}")
    print(f"{'='*70}")
    
    # Statistics by number of decisions activated
    print(f"\nBy number of decisions activated:")
    for n in range(4):  # 0, 1, 2, 3 decisions
        count = (states_t['total_decisions'] == n).sum()
        pct = count / total_households * 100
        print(f"  {n} decision(s):  {count:3d} households ({pct:5.1f}%)")
    
    # Detailed combination breakdown
    print(f"\nDetailed combinations:")
    
    # No decisions
    none = ((states_t['repair'] == 0) & 
            (states_t['vacant'] == 0) & 
            (states_t['sell'] == 0)).sum()
    print(f"  (0,0,0) None:           {none:3d}")
    
    # Single decisions
    repair_only = ((states_t['repair'] == 1) & 
                   (states_t['vacant'] == 0) & 
                   (states_t['sell'] == 0)).sum()
    vacant_only = ((states_t['repair'] == 0) & 
                   (states_t['vacant'] == 1) & 
                   (states_t['sell'] == 0)).sum()
    sell_only = ((states_t['repair'] == 0) & 
                 (states_t['vacant'] == 0) & 
                 (states_t['sell'] == 1)).sum()
    
    print(f"  (1,0,0) Repair only:    {repair_only:3d}")
    print(f"  (0,1,0) Vacant only:    {vacant_only:3d}")
    print(f"  (0,0,1) Sell only:      {sell_only:3d}")
    
    # Two decisions
    repair_vacant = ((states_t['repair'] == 1) & 
                     (states_t['vacant'] == 1) & 
                     (states_t['sell'] == 0)).sum()
    repair_sell = ((states_t['repair'] == 1) & 
                   (states_t['vacant'] == 0) & 
                   (states_t['sell'] == 1)).sum()
    vacant_sell = ((states_t['repair'] == 0) & 
                   (states_t['vacant'] == 1) & 
                   (states_t['sell'] == 1)).sum()
    
    if repair_vacant + repair_sell + vacant_sell > 0:
        print(f"  (1,1,0) Repair+Vacant:  {repair_vacant:3d}")
        print(f"  (1,0,1) Repair+Sell:    {repair_sell:3d}")
        print(f"  (0,1,1) Vacant+Sell:    {vacant_sell:3d}")
    
    # Three decisions
    all_three = ((states_t['repair'] == 1) & 
                 (states_t['vacant'] == 1) & 
                 (states_t['sell'] == 1)).sum()
    
    if all_three > 0:
        print(f"  (1,1,1) All three:      {all_three:3d}")
    
    print(f"{'='*70}\n")
    
    # Return summary dict for potential further use
    return {
        'timestep': timestep,
        'total_households': total_households,
        'no_decision': none,
        'single_decision': repair_only + vacant_only + sell_only,
        'two_decisions': repair_vacant + repair_sell + vacant_sell,
        'three_decisions': all_three,
        'repair_only': repair_only,
        'vacant_only': vacant_only,
        'sell_only': sell_only,
        'repair_vacant': repair_vacant,
        'repair_sell': repair_sell,
        'vacant_sell': vacant_sell,
        'all_three': all_three
    }

def run_all_abm_scenarios(base_output_path: str = './dataset'):
    """
    Run all predefined ABM scenarios and create summary (single run per scenario)

    Args:
        base_output_path: Base path for all outputs
    
    Returns:
        List of statistics dictionaries for all scenarios
    """
    print("\n" + "="*70)
    print("RUNNING ALL ABM SCENARIOS (SINGLE RUN MODE)")
    print("="*70)
    print(f"Number of scenarios: {len(ALL_ABM_SCENARIOS)}")
    print(f"Base output path: {base_output_path}")

    # Create base output directory
    os.makedirs(base_output_path, exist_ok=True)

    # Run each scenario
    all_statistics = []
    for i, config in enumerate(ALL_ABM_SCENARIOS, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Scenario {i}/{len(ALL_ABM_SCENARIOS)}: {config.name}")
        print(f"{'#'*70}")
    
        try:
            stats = run_abm_scenario(config, base_output_path)
            all_statistics.append(stats)
        except Exception as e:
            print(f"\nERROR in scenario {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary Excel
    if all_statistics:
        summary_path = os.path.join(base_output_path, 'dataset_summary.xlsx')
        create_abm_summary_excel(all_statistics, summary_path)
    
        print("\n" + "="*70)
        print("ALL SCENARIOS COMPLETED")
        print("="*70)
        print(f"Total scenarios processed: {len(all_statistics)}/{len(ALL_ABM_SCENARIOS)}")
        print(f"Summary saved to: {summary_path}")
    else:
        print("\nNo scenarios completed successfully!")

    return all_statistics

def run_all_abm_scenarios_multiseed(base_output_path: str = './dataset',
                                    n_runs: int = 10,
                                    save_full_data: bool = False):
    """
    Run all predefined ABM scenarios with multiple seeds

    Args:
        base_output_path: Base path for all outputs
        n_runs: Number of runs per scenario with different seeds
        save_full_data: If True, save complete datasets for each run
    
    Returns:
        List of aggregated statistics for all scenarios
    """
    print("\n" + "="*70)
    print("RUNNING ALL ABM SCENARIOS (MULTI-SEED MODE)")
    print("="*70)
    print(f"Number of scenarios: {len(ALL_ABM_SCENARIOS)}")
    print(f"Runs per scenario: {n_runs}")
    print(f"Save full data: {save_full_data}")
    print(f"Base output path: {base_output_path}")

    # Create base output directory
    os.makedirs(base_output_path, exist_ok=True)

    # Run each scenario with multiple seeds
    all_aggregated_stats = []
    for i, config in enumerate(ALL_ABM_SCENARIOS, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Scenario {i}/{len(ALL_ABM_SCENARIOS)}: {config.name}")
        print(f"{'#'*70}")
    
        try:
            agg_stats = run_abm_scenario_multiseed(
                config, 
                base_output_path, 
                n_runs=n_runs,
                save_full_data=save_full_data
            )
            all_aggregated_stats.append(agg_stats)
        except Exception as e:
            print(f"\nERROR in scenario {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary
    if all_aggregated_stats:
        print("\n" + "="*70)
        print("ALL MULTI-SEED SCENARIOS COMPLETED")
        print("="*70)
        print(f"Total scenarios processed: {len(all_aggregated_stats)}/{len(ALL_ABM_SCENARIOS)}")
        print(f"Total simulations run: {len(all_aggregated_stats) * n_runs}")
    
        # Save overall summary
        summary_path = os.path.join(base_output_path, 'multiseed_overall_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump({
                'n_scenarios': len(all_aggregated_stats),
                'n_runs_per_scenario': n_runs,
                'scenarios': [s['scenario_name'] for s in all_aggregated_stats],
                'aggregated_stats': all_aggregated_stats
            }, f, indent=2)
        print(f"Overall summary saved to: {summary_path}")
    else:
        print("\nNo scenarios completed successfully!")

    return all_aggregated_stats

def run_single_abm_scenario_by_name(scenario_name: str, base_output_path: str = './dataset'):
    """
    Run a single ABM scenario by name

    Args:
        scenario_name: Name of the scenario (e.g., 'ABM_Sparse_LowSeed')
        base_output_path: Base path for output
    
    Returns:
        Statistics dictionary
    """
    config = get_abm_scenario(scenario_name)
    return run_abm_scenario(config, base_output_path)

def run_custom_abm_scenario(custom_config: ABMScenarioConfig, base_output_path: str = './dataset'):
    """
    Run a custom ABM scenario with user-defined configuration

    Args:
        custom_config: Custom ABMScenarioConfig object
        base_output_path: Base path for output
    
    Returns:
        Statistics dictionary
    """
    return run_abm_scenario(custom_config, base_output_path)

if __name__ == "__main__":
    # Get the project root directory (parent of current script directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_path = os.path.join(parent_dir, 'dataset')

    print("ABM Data Generation System")
    print("="*70)
    print("\nAvailable scenarios:")
    for i, scenario in enumerate(ALL_ABM_SCENARIOS, 1):
        print(f"  {i}. {scenario.name}: {scenario.description}")

    # ============================================================
    # CHOOSE MODE: Single-run or Multi-seed
    # ============================================================

    # MODE 1: Single run per scenario (original behavior)
    # Uncomment to use:
    print("\n" + "="*70)
    print("Starting single-run dataset generation...")
    print("="*70)
    all_stats = run_all_abm_scenarios(base_output_path=output_path)

    # # MODE 2: Multi-seed runs (10 runs per scenario) - NEW
    # # Uncomment to use:
    # print("\n" + "="*70)
    # print("Starting multi-seed dataset generation...")
    # print("="*70)
    # all_agg_stats = run_all_abm_scenarios_multiseed(
    #     base_output_path=output_path,
    #     n_runs=10,
    #     save_full_data=False  # Set to True if you need all CSVs
    # )

    # Option 3: Run single scenario with multiple seeds (uncomment to use)
    # agg_stats = run_abm_scenario_multiseed(
    #     config=ALL_ABM_SCENARIOS[0],
    #     base_output_path=output_path,
    #     n_runs=10,
    #     save_full_data=True
    # )

    print("\n" + "="*70)
    print("ABM DATA GENERATION COMPLETED SUCCESSFULLY!")
    print("="*70)

