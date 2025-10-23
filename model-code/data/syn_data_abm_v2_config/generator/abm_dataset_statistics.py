# abm_dataset_statistics.py
import pandas as pd
import numpy as np
import json
import networkx as nx
from typing import Dict, Any
from abm_config import ABMScenarioConfig

def calculate_network_metrics(network_df: pd.DataFrame, timestep: int) -> Dict[str, Any]:
    """
    Calculate comprehensive network metrics for a given timestep
    
    Args:
        network_df: DataFrame with columns [household_id_1, household_id_2, timestep, link_type]
        timestep: The timestep to analyze
        
    Returns:
        Dictionary with network metrics
    """
    # Filter to specific timestep
    df_t = network_df[network_df['timestep'] == timestep].copy()
    
    if len(df_t) == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0.0,
            'n_connected_components': 0,
            'degree_centrality_mean': 0.0,
            'degree_centrality_max': 0.0,
            'degree_centrality_std': 0.0,
            'betweenness_centrality_mean': 0.0,
            'betweenness_centrality_max': 0.0,
            'closeness_centrality_mean': 0.0,
            'closeness_centrality_max': 0.0,
            'clustering_coefficient_global': 0.0,
            'clustering_coefficient_mean': 0.0
        }
    
    # Build networkx graph
    G = nx.Graph()
    for _, row in df_t.iterrows():
        G.add_edge(row['household_id_1'], row['household_id_2'], 
                   link_type=row['link_type'])
    
    # Basic metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    n_components = nx.number_connected_components(G)
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    degree_values = list(degree_cent.values())
    degree_mean = np.mean(degree_values) if degree_values else 0.0
    degree_max = np.max(degree_values) if degree_values else 0.0
    degree_std = np.std(degree_values) if degree_values else 0.0
    
    # Betweenness centrality (can be slow for large networks)
    if num_nodes > 0 and num_nodes <= 500:  # Only calculate if reasonable size
        between_cent = nx.betweenness_centrality(G)
        between_values = list(between_cent.values())
        between_mean = np.mean(between_values) if between_values else 0.0
        between_max = np.max(between_values) if between_values else 0.0
    else:
        between_mean = 0.0
        between_max = 0.0
    
    # Closeness centrality
    if num_nodes > 0 and nx.is_connected(G):
        close_cent = nx.closeness_centrality(G)
        close_values = list(close_cent.values())
        close_mean = np.mean(close_values) if close_values else 0.0
        close_max = np.max(close_values) if close_values else 0.0
    else:
        # For disconnected graphs, calculate per component
        close_values = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(component) > 1:
                close_cent_comp = nx.closeness_centrality(subgraph)
                close_values.extend(close_cent_comp.values())
        close_mean = np.mean(close_values) if close_values else 0.0
        close_max = np.max(close_values) if close_values else 0.0
    
    # Clustering coefficient
    if num_nodes > 0:
        clustering_global = nx.transitivity(G)
        clustering_local = nx.clustering(G)
        clustering_mean = np.mean(list(clustering_local.values())) if clustering_local else 0.0
    else:
        clustering_global = 0.0
        clustering_mean = 0.0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': round(density, 4),
        'n_connected_components': n_components,
        'degree_centrality_mean': round(degree_mean, 4),
        'degree_centrality_max': round(degree_max, 4),
        'degree_centrality_std': round(degree_std, 4),
        'betweenness_centrality_mean': round(between_mean, 4),
        'betweenness_centrality_max': round(between_max, 4),
        'closeness_centrality_mean': round(close_mean, 4),
        'closeness_centrality_max': round(close_max, 4),
        'clustering_coefficient_global': round(clustering_global, 4),
        'clustering_coefficient_mean': round(clustering_mean, 4)
    }

def calculate_abm_statistics(household_states: pd.DataFrame,
                             ground_truth_network: pd.DataFrame,
                             config: ABMScenarioConfig,
                             missing_stats: Dict = None) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for the ABM generated dataset
    
    Args:
        household_states: DataFrame with columns [household_id, timestep, repair, vacant, sell]
        ground_truth_network: DataFrame with columns [household_id_1, household_id_2, timestep, link_type]
        config: ABMScenarioConfig object with scenario parameters
        missing_stats: Optional dictionary with missing mechanism statistics
        
    Returns:
        Dictionary containing all statistics
    """
    T = config.n_timesteps
    
    # Get states at t=0 and t=T-1
    states_t0 = household_states[household_states['timestep'] == 0]
    states_tT = household_states[household_states['timestep'] == T-1]
    
    # Get network at t=0 and t=T-1
    network_t0 = ground_truth_network[ground_truth_network['timestep'] == 0]
    network_tT = ground_truth_network[ground_truth_network['timestep'] == T-1]
    
    # Calculate decision statistics
    repair_t0 = int((states_t0['repair'] == 1).sum())
    vacant_t0 = int((states_t0['vacant'] == 1).sum())
    sell_t0 = int((states_t0['sell'] == 1).sum())

    repair_tT = int((states_tT['repair'] == 1).sum())
    vacant_tT = int((states_tT['vacant'] == 1).sum())
    sell_tT = int((states_tT['sell'] == 1).sum())
    
    total_households = len(states_t0)
    
    # Calculate network statistics
    bonding_t0 = int((network_t0['link_type'] == 1).sum())
    bridging_t0 = int((network_t0['link_type'] == 2).sum())
    bonding_tT = int((network_tT['link_type'] == 1).sum())
    bridging_tT = int((network_tT['link_type'] == 2).sum())
    
    # Calculate average degree
    total_links_t0 = len(network_t0)
    total_links_tT = len(network_tT)
    
    avg_degree_t0 = (2 * total_links_t0) / total_households if total_households > 0 else 0
    avg_degree_tT = (2 * total_links_tT) / total_households if total_households > 0 else 0
    
    # Calculate actual seeding ratio at t=0
    total_seeds_t0 = repair_t0 + vacant_t0 + sell_t0
    actual_seed_ratio = total_seeds_t0 / total_households if total_households > 0 else 0
    
    # Calculate actual bonding ratio at t=0
    actual_bonding_ratio_t0 = bonding_t0 / total_links_t0 if total_links_t0 > 0 else 0
    
    # NEW: Calculate network metrics for t=0 and t=T
    print("  Calculating network metrics for t=0...")
    network_metrics_t0 = calculate_network_metrics(ground_truth_network, 0)
    print("  Calculating network metrics for t=T...")
    network_metrics_tT = calculate_network_metrics(ground_truth_network, T-1)
    
    # Compile all statistics
    stats = {
        'scenario_name': config.name,
        'description': config.description,
        'total_households': total_households,
        'time_horizon': T,
        
        # Configuration targets
        'target_avg_degree': config.target_avg_degree,
        'target_seed_ratio': config.target_seed_ratio,
        'target_bonding_ratio': config.target_bonding_ratio,
        
        # Network statistics
        'bonding_links_t0': bonding_t0,
        'bridging_links_t0': bridging_t0,
        'bonding_links_tT': bonding_tT,
        'bridging_links_tT': bridging_tT,
        'total_links_t0': total_links_t0,
        'total_links_tT': total_links_tT,
        'avg_degree_t0': round(avg_degree_t0, 2),
        'avg_degree_tT': round(avg_degree_tT, 2),
        'actual_bonding_ratio_t0': round(actual_bonding_ratio_t0, 3),
        
        # NEW: Network metrics at t=0
        'network_metrics_t0': network_metrics_t0,
        
        # NEW: Network metrics at t=T
        'network_metrics_tT': network_metrics_tT,
        
        # Decision statistics (counts)
        'repair_t0': repair_t0,
        'vacant_t0': vacant_t0,
        'sell_t0': sell_t0,
        'repair_tT': repair_tT,
        'vacant_tT': vacant_tT,
        'sell_tT': sell_tT,
        'total_seeds_t0': total_seeds_t0,
        
        # Decision statistics (percentages)
        'repair_pct_t0': round(100 * repair_t0 / total_households, 1),
        'vacant_pct_t0': round(100 * vacant_t0 / total_households, 1),
        'sell_pct_t0': round(100 * sell_t0 / total_households, 1),
        'repair_pct_tT': round(100 * repair_tT / total_households, 1),
        'vacant_pct_tT': round(100 * vacant_tT / total_households, 1),
        'sell_pct_tT': round(100 * sell_tT / total_households, 1),
        'actual_seed_ratio': round(actual_seed_ratio, 3),
        
        # Role distribution (from config)
        'early_adopter_ratio': config.early_adopter_ratio,
        'social_follower_ratio': config.social_follower_ratio,
        'resistant_ratio': config.resistant_ratio,
        'isolated_ratio': config.isolated_ratio,
        
        # Missing mechanism statistics
        'missing_stats': missing_stats if missing_stats else {}
    }
    
    return stats

def print_abm_statistics(stats: Dict[str, Any]):
    """Print dataset statistics in a formatted way"""
    print(f"\n{'='*70}")
    print(f"ABM Dataset Statistics: {stats['scenario_name']}")
    print(f"{'='*70}")
    print(f"Description: {stats['description']}")
    print(f"Total Households: {stats['total_households']}")
    print(f"Time Horizon: {stats['time_horizon']}")
    print()
    
    print("Configuration Targets:")
    print(f"  Target avg degree: {stats['target_avg_degree']}")
    print(f"  Target seed ratio: {stats['target_seed_ratio']:.1%}")
    print(f"  Target bonding ratio: {stats['target_bonding_ratio']:.1%}")
    print()
    
    print("Role Distribution:")
    print(f"  Early adopter: {stats['early_adopter_ratio']:.1%}")
    print(f"  Social follower: {stats['social_follower_ratio']:.1%}")
    print(f"  Resistant: {stats['resistant_ratio']:.1%}")
    print(f"  Isolated: {stats['isolated_ratio']:.1%}")
    print()
    
    print("Network Statistics:")
    print(f"  Links at t=0: {stats['total_links_t0']} total "
          f"({stats['bonding_links_t0']} bonding, {stats['bridging_links_t0']} bridging)")
    print(f"  Links at t=T: {stats['total_links_tT']} total "
          f"({stats['bonding_links_tT']} bonding, {stats['bridging_links_tT']} bridging)")
    print(f"  Avg degree at t=0: {stats['avg_degree_t0']} (target: {stats['target_avg_degree']})")
    print(f"  Avg degree at t=T: {stats['avg_degree_tT']}")
    print(f"  Actual bonding ratio at t=0: {stats['actual_bonding_ratio_t0']:.1%} "
          f"(target: {stats['target_bonding_ratio']:.1%})")
    print()
    
    # NEW: Print network metrics
    if 'network_metrics_t0' in stats:
        print("Network Metrics at t=0:")
        m = stats['network_metrics_t0']
        print(f"  Nodes: {m['num_nodes']}, Edges: {m['num_edges']}, Density: {m['density']}")
        print(f"  Connected components: {m['n_connected_components']}")
        print(f"  Degree centrality: mean={m['degree_centrality_mean']}, max={m['degree_centrality_max']}, std={m['degree_centrality_std']}")
        print(f"  Betweenness centrality: mean={m['betweenness_centrality_mean']}, max={m['betweenness_centrality_max']}")
        print(f"  Closeness centrality: mean={m['closeness_centrality_mean']}, max={m['closeness_centrality_max']}")
        print(f"  Clustering coefficient: global={m['clustering_coefficient_global']}, mean={m['clustering_coefficient_mean']}")
        print()
    
    if 'network_metrics_tT' in stats:
        print("Network Metrics at t=T:")
        m = stats['network_metrics_tT']
        print(f"  Nodes: {m['num_nodes']}, Edges: {m['num_edges']}, Density: {m['density']}")
        print(f"  Connected components: {m['n_connected_components']}")
        print(f"  Degree centrality: mean={m['degree_centrality_mean']}, max={m['degree_centrality_max']}, std={m['degree_centrality_std']}")
        print(f"  Betweenness centrality: mean={m['betweenness_centrality_mean']}, max={m['betweenness_centrality_max']}")
        print(f"  Closeness centrality: mean={m['closeness_centrality_mean']}, max={m['closeness_centrality_max']}")
        print(f"  Clustering coefficient: global={m['clustering_coefficient_global']}, mean={m['clustering_coefficient_mean']}")
        print()
    
    print("Decision Statistics at t=0:")
    print(f"  Repair: {stats['repair_t0']} ({stats['repair_pct_t0']}%)")
    print(f"  Vacant: {stats['vacant_t0']} ({stats['vacant_pct_t0']}%)")
    print(f"  Sell: {stats['sell_t0']} ({stats['sell_pct_t0']}%)")
    print(f"  Total seeds: {stats['total_seeds_t0']} ({stats['actual_seed_ratio']:.1%}, "
          f"target: {stats['target_seed_ratio']:.1%})")
    print()
    
    print("Decision Statistics at t=T:")
    print(f"  Repair: {stats['repair_tT']} ({stats['repair_pct_tT']}%)")
    print(f"  Vacant: {stats['vacant_tT']} ({stats['vacant_pct_tT']}%)")
    print(f"  Sell: {stats['sell_tT']} ({stats['sell_pct_tT']}%)")
    print()
    
    # Print missing mechanisms info if available
    if 'missing_stats' in stats and stats['missing_stats']:
        ms = stats['missing_stats']
        print("Missing Mechanisms:")
        print(f"  Structural missing rates:")
        print(f"    - Bonding: {ms['config_structural_bonding']*100:.0f}%")
        print(f"    - Bridging: {ms['config_structural_bridging']*100:.0f}%")
        print(f"  Temporal missing rates:")
        print(f"    - Bonding: {ms['config_temporal_bonding']*100:.0f}%")
        print(f"    - Bridging: {ms['config_temporal_bridging']*100:.0f}%")
        print(f"  Overall missing rate: {ms['overall_missing_rate']}%")
        print(f"  Unique edges missing: {ms['unique_edges_missing_rate']}%")
        print()
    
    # Print deviations from targets
    degree_deviation = abs(stats['avg_degree_t0'] - stats['target_avg_degree']) / stats['target_avg_degree']
    seed_deviation = abs(stats['actual_seed_ratio'] - stats['target_seed_ratio']) / stats['target_seed_ratio']
    bonding_deviation = abs(stats['actual_bonding_ratio_t0'] - stats['target_bonding_ratio']) / stats['target_bonding_ratio'] if stats['target_bonding_ratio'] > 0 else 0
    
    print("Target Achievement:")
    print(f"  Avg degree deviation: {degree_deviation:.1%}")
    print(f"  Seed ratio deviation: {seed_deviation:.1%}")
    print(f"  Bonding ratio deviation: {bonding_deviation:.1%}")
    print(f"{'='*70}\n")

def save_abm_statistics(stats: Dict[str, Any], output_folder: str):
    """Save statistics to JSON file"""
    with open(f"{output_folder}/statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

def create_abm_summary_excel(all_statistics: list, output_path: str):
    """Create Excel summary of all ABM scenario statistics"""
    summary_df = pd.DataFrame(all_statistics)
    
    # Reorder columns for better readability
    column_order = [
        'scenario_name', 'description', 'total_households', 'time_horizon',
        # Configuration
        'target_avg_degree', 'target_seed_ratio', 'target_bonding_ratio',
        # Role distribution
        'early_adopter_ratio', 'social_follower_ratio', 'resistant_ratio', 'isolated_ratio',
        # Network stats
        'total_links_t0', 'bonding_links_t0', 'bridging_links_t0',
        'total_links_tT', 'bonding_links_tT', 'bridging_links_tT',
        'avg_degree_t0', 'avg_degree_tT', 'actual_bonding_ratio_t0',
        # Decision stats
        'total_seeds_t0', 'actual_seed_ratio',
        'repair_t0', 'vacant_t0', 'sell_t0',
        'repair_tT', 'vacant_tT', 'sell_tT',
        'repair_pct_t0', 'vacant_pct_t0', 'sell_pct_t0',
        'repair_pct_tT', 'vacant_pct_tT', 'sell_pct_tT'
    ]
    
    # Only include columns that exist in the dataframe
    existing_columns = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[existing_columns]
    
    summary_df.to_excel(output_path, index=False)
    print(f"\nABM Summary Excel saved to: {output_path}")
    
    return summary_df