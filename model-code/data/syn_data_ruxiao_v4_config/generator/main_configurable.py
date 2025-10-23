# main_configurable.py - WITH p_self and p_ji recording
import pdb
import pygeohash as pgh
from generate_household_features import generate_household_features
from generate_household_states import generate_T0_states_with_config, update_full_states_one_step
import pandas as pd
from household_features_function import compute_similarity, compute_interaction_potential
from links_updates_fun import generate_initial_links, compute_p_self, compute_p_ji_linear, update_link_matrix_one_step
import numpy as np
from tqdm import tqdm
import os
import warnings
from config import ScenarioConfig, ALL_SCENARIOS, get_scenario
from config import fill_missing_defaults
from blocking_modes import apply_blocking_mode
from dataset_statistics import calculate_dataset_statistics, print_statistics, save_statistics, create_summary_excel

warnings.filterwarnings("ignore")


def run_scenario(config: ScenarioConfig, base_data_path):
    """
    Run simulation for a single scenario configuration
    
    Args:
        config: ScenarioConfig object
        base_data_path: Path to your base data directory
        
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*50}")
    print(f"Running scenario: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*50}")
    
    config = fill_missing_defaults(config)

    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    
    # Use config parameters instead of hardcoded values
    alpha = config.alpha_bonding
    beta = config.beta_bridging
    gamma = config.gamma_decay
    L = config.L
    state_dims = ['vacancy_state', 'repair_state', 'sales_state']
    T = config.time_horizon
    p_block = config.p_block

    # Save Results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(os.path.dirname(current_dir), 'dataset', config.name)
    print(f"Saving results to: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Read real Household Nodes
    df_ori = pd.read_csv(os.path.join(base_data_path, 'household_swinMaryland_20190101.csv'))
    df_ori = df_ori[['home_1', 'home_2', 'home_1_number', 'home_2_number']]

    # Extract home and their locations
    home1 = df_ori[['home_1']].rename(columns={'home_1': 'home'})
    home2 = df_ori[['home_2']].rename(columns={'home_2': 'home'})

    # Merge and remove duplicates
    house_df = pd.concat([home1, home2], ignore_index=True).drop_duplicates(subset='home')
    house_df = house_df.dropna()
    house_df = house_df[:config.n_households]
    
    # Household Features Generation
    house_df_with_features = generate_household_features(house_df)

    # T=0 Household States Generation
    house_states = generate_T0_states_with_config(house_df_with_features, T, config)

    # T=0 Household Similarity and Interaction Potential
    similarity_df = compute_similarity(house_df_with_features)
    interaction_df = compute_interaction_potential(house_df_with_features, house_states, t=0)

    # T=0 Links Generation
    initial_links_df = generate_initial_links(similarity_df, interaction_df, config, alpha_0=alpha, beta_0=beta, output_folder=folder_path)

    # Store link matrices for each t
    link_snapshots = {0: initial_links_df.copy()}

    # ===== NEW: Initialize storage for p_self and p_ji values =====
    p_self_all_values = []  # Store all individual p_self values
    p_ji_all_values = []    # Store all individual p_ji values
    # ===============================================================

    # Main simulation loop
    for t in tqdm(range(T - 1)):
        print(f'--- sim step  {t}  →  {t+1} ---')

        # State-update for each dimension k ∈ {0,1,2}
        for k, k_col in enumerate(state_dims):
            # Compute p_self and p_ji for current (t, k)
            p_self = compute_p_self(
                house_df_with_features.set_index('home'),
                house_states,
                t=t/T,
                k=k,
                L=L
            )

            p_ji = compute_p_ji_linear(
                link_snapshots[t],
                house_df_with_features,
                house_states,
                t=t/T,
                k=k,
                L=L
            )
            
            # ===== NEW: Record all individual p_self values =====
            for household_id in p_self.index:
                p_self_all_values.append({
                    'time_step': t,
                    'dimension': k,
                    'dimension_name': state_dims[k],
                    'household_id': household_id,
                    'p_self_value': p_self[household_id]
                })
            
            # ===== NEW: Record all individual p_ji values =====
            for household_i in p_ji.index:
                for household_j in p_ji.columns:
                    p_ji_all_values.append({
                        'time_step': t,
                        'dimension': k,
                        'dimension_name': state_dims[k],
                        'household_i': household_i,
                        'household_j': household_j,
                        'p_ji_value': p_ji.loc[household_i, household_j]
                    })
            # ====================================================
            
            # Update states
            house_states, _ = update_full_states_one_step(
                house_df_with_features,
                house_states,
                p_self,
                p_ji,
                link_snapshots[t],
                t=t,
                k=k,
                config=config
            )

        # Link-transition G_t → G_{t+1}
        sim_t = compute_similarity(house_df_with_features)
        link_next = update_link_matrix_one_step(
            sim_t,
            interaction_df,
            link_snapshots[t],
            house_states,
            t=t,
            alpha_bonding=alpha,
            beta_form=beta,
            gamma=gamma
        )

        link_snapshots[t + 1] = link_next

    # Process results
    link_records = []
    for tt, g_df in link_snapshots.items():
        g_upper = g_df.where(np.triu(np.ones(g_df.shape), k=1).astype(bool))
        g_long = g_upper.stack().reset_index()
        g_long.columns = ['home_i', 'home_j', 'link_type']
        g_long['time_step'] = tt
        link_records.append(g_long)

    links_long_df = pd.concat(link_records, ignore_index=True)
    house_states = house_states.sort_values(['time', 'home']).reset_index(drop=True)

    print("Simulation finished.")
    count_ones = (initial_links_df == 1).sum().sum()
    print(f"Number of bonding links: {count_ones/2}")

    # Prepare data for saving
    links_long_df = links_long_df.rename(columns={'home_i': 'household_id_1','home_j': 'household_id_2','time_step': 'time_step'})
    house_df_with_features = house_df_with_features.rename(columns={'home': 'household_id'})

    links_long_df = links_long_df[links_long_df['link_type'] != 0]

    # Apply blocking mode based on configuration
    print(f"Applying {config.blocking_mode.upper()} blocking with p={config.p_block}")
    blocked_df = apply_blocking_mode(links_long_df, config.blocking_mode, config.p_block)
    house_df['latitude'], house_df['longitude'] = zip(*house_df['home'].map(pgh.decode))
    house_df = house_df.rename(columns={'home': 'household_id'})

    # Save datasets
    house_states.to_csv(folder_path+'/household_states_raw.csv', index=False)
    links_long_df.to_csv(folder_path+'/ground_truth_network_raw.csv', index=False)
    blocked_df.to_csv(folder_path+'/observed_network_raw.csv', index=False)
    house_df.to_csv(folder_path+'/household_locations_raw.csv', index=False)
    house_df_with_features.to_csv(folder_path+'/household_features_raw.csv', index=False)
    similarity_df.to_csv(folder_path+"/similarity_df_raw.csv")

    # ===== NEW: Save all individual p_self and p_ji values =====
    p_self_all_df = pd.DataFrame(p_self_all_values)
    p_ji_all_df = pd.DataFrame(p_ji_all_values)

    p_self_all_df.to_csv(folder_path+"/p_self_all_values.csv", index=False)
    p_ji_all_df.to_csv(folder_path+"/p_ji_all_values.csv", index=False)

    print(f"p_self and p_ji values saved to {folder_path}")
    # ===========================================================

    # Calculate and save statistics
    stats = calculate_dataset_statistics(house_states, links_long_df, config)
    save_statistics(stats, folder_path)
    print_statistics(stats)
    
    print(f"Results saved to: {folder_path}")
    return stats

def run_all_scenarios():
    """Run all predefined scenarios and create summary"""
    all_statistics = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for config in ALL_SCENARIOS:
        config = fill_missing_defaults(config)
        stats = run_scenario(config, base_data_path=os.path.join(os.path.dirname(current_dir), 'generator'))
        all_statistics.append(stats)
    
    # Create summary Excel
    summary_path = current_dir = os.path.dirname(current_dir)+'/dataset/dataset_summary.xlsx'
    create_summary_excel(all_statistics, summary_path)
    
    return all_statistics

def run_single_scenario_by_name(scenario_name: str):
    """Run a single scenario by name"""
    config = get_scenario(scenario_name)
    config = fill_missing_defaults(config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return run_scenario(config, base_data_path=os.path.join(os.path.dirname(current_dir), 'generator'))

if __name__ == "__main__":
    # Option 1: Run all scenarios
    print("Running all scenarios...")
    all_stats = run_all_scenarios()
    
    # Option 2: Run single scenario (uncomment to use)
    # stats = run_single_scenario_by_name("G1_Sparse_LowSeed")
    
    # Option 3: Run specific scenarios (uncomment to use)
    # for scenario_name in ["G1_Sparse_LowSeed", "G2_Sparse_HighSeed"]:
    #     stats = run_single_scenario_by_name(scenario_name)