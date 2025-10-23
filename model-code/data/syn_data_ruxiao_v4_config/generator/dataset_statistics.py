# statistics.py
import pandas as pd
import json
import networkx as nx
import numpy as np
from typing import Dict, Any
from config import ScenarioConfig

def calculate_network_features(links_df: pd.DataFrame, households: list) -> Dict[str, float]:
    """
    Calculate network features using NetworkX
    
    Args:
        links_df: DataFrame with link information at a specific timestep
        households: List of all household IDs
        
    Returns:
        Dictionary with network features
    """
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(households)
    
    # Add edges (only if link_type > 0)
    for _, row in links_df.iterrows():
        if row['link_type'] > 0:
            G.add_edge(row['household_id_1'], row['household_id_2'])
    
    # Calculate features
    features = {}
    
    # Basic connectivity
    features['num_nodes'] = G.number_of_nodes()
    features['num_edges'] = G.number_of_edges()
    features['density'] = nx.density(G)
    
    # Degree statistics
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    features['avg_degree'] = np.mean(degree_values) if degree_values else 0
    features['max_degree'] = np.max(degree_values) if degree_values else 0
    features['min_degree'] = np.min(degree_values) if degree_values else 0
    
    # Centrality measures (only for connected components with >2 nodes)
    if G.number_of_edges() > 0:
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        features['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
        features['max_degree_centrality'] = np.max(list(degree_cent.values()))
        
        # Betweenness centrality (expensive, only for smaller graphs)
        if len(households) <= 500:
            between_cent = nx.betweenness_centrality(G)
            features['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
            features['max_betweenness_centrality'] = np.max(list(between_cent.values()))
        else:
            features['avg_betweenness_centrality'] = None
            features['max_betweenness_centrality'] = None
        
        # Closeness centrality (only for connected graphs)
        if nx.is_connected(G):
            close_cent = nx.closeness_centrality(G)
            features['avg_closeness_centrality'] = np.mean(list(close_cent.values()))
            features['max_closeness_centrality'] = np.max(list(close_cent.values()))
        else:
            features['avg_closeness_centrality'] = None
            features['max_closeness_centrality'] = None
    else:
        features['avg_degree_centrality'] = 0
        features['max_degree_centrality'] = 0
        features['avg_betweenness_centrality'] = 0
        features['max_betweenness_centrality'] = 0
        features['avg_closeness_centrality'] = None
        features['max_closeness_centrality'] = None
    
    # Clustering coefficient
    features['avg_clustering'] = nx.average_clustering(G)
    
    # Connected components
    features['num_components'] = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len) if G.number_of_edges() > 0 else set()
    features['largest_component_size'] = len(largest_cc)
    features['largest_component_ratio'] = len(largest_cc) / len(households) if len(households) > 0 else 0
    
    return features

def calculate_dataset_statistics(house_states: pd.DataFrame, 
                               links_long_df: pd.DataFrame,
                               config: ScenarioConfig) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for the generated dataset
    
    Args:
        house_states: DataFrame with household states over time
        links_long_df: DataFrame with network links over time  
        config: Configuration object with scenario parameters
        
    Returns:
        Dictionary containing all required statistics
    """
    T = config.time_horizon
    
    # Get states at t=0 and t=T-1
    states_t0 = house_states[house_states['time'] == 0]
    states_tT = house_states[house_states['time'] == T-1]
    
    # Get links at t=0 and t=T-1
    links_t0 = links_long_df[links_long_df['time_step'] == 0]
    links_tT = links_long_df[links_long_df['time_step'] == T-1]
    
    # Calculate decision statistics
    sell_t0 = int(states_t0['sales_state'].sum())
    vacant_t0 = int(states_t0['vacancy_state'].sum())
    repair_t0 = int(states_t0['repair_state'].sum())
    
    sell_tT = int(states_tT['sales_state'].sum())
    vacant_tT = int(states_tT['vacancy_state'].sum())
    repair_tT = int(states_tT['repair_state'].sum())
    
    total_households = len(states_t0)
    all_households = states_t0['home'].tolist()
    
    # Calculate link statistics
    bonding_t0 = int((links_t0['link_type'] == 1).sum())
    bridging_t0 = int((links_t0['link_type'] == 2).sum())
    bonding_tT = int((links_tT['link_type'] == 1).sum())
    bridging_tT = int((links_tT['link_type'] == 2).sum())
    
    # Calculate average degree (each link connects 2 nodes)
    total_links_t0 = len(links_t0[links_t0['link_type'] > 0])
    total_links_tT = len(links_tT[links_tT['link_type'] > 0])
    
    avg_degree_t0 = (2 * total_links_t0) / total_households if total_households > 0 else 0
    avg_degree_tT = (2 * total_links_tT) / total_households if total_households > 0 else 0
    
    # Calculate network features at t=0 and t=T-1
    print("Calculating network features at t=0...")
    network_features_t0 = calculate_network_features(links_t0, all_households)
    print("Calculating network features at t=T-1...")
    network_features_tT = calculate_network_features(links_tT, all_households)
    
    # Compile all statistics
    stats = {
        'scenario_name': config.name,
        'description': config.description,
        'total_households': total_households,
        'blocking_mode': config.blocking_mode,
        'time_horizon': T,
        
        # Network statistics
        'bonding_links_t0': bonding_t0,
        'bridging_links_t0': bridging_t0,
        'bonding_links_tT': bonding_tT,
        'bridging_links_tT': bridging_tT,
        'avg_degree_t0': round(avg_degree_t0, 2),
        'avg_degree_tT': round(avg_degree_tT, 2),
        
        # Network features at t=0
        'density_t0': round(network_features_t0['density'], 4),
        'avg_degree_centrality_t0': round(network_features_t0['avg_degree_centrality'], 4),
        'max_degree_centrality_t0': round(network_features_t0['max_degree_centrality'], 4),
        'avg_betweenness_centrality_t0': round(network_features_t0['avg_betweenness_centrality'], 4) if network_features_t0['avg_betweenness_centrality'] is not None else None,
        'max_betweenness_centrality_t0': round(network_features_t0['max_betweenness_centrality'], 4) if network_features_t0['max_betweenness_centrality'] is not None else None,
        'avg_clustering_t0': round(network_features_t0['avg_clustering'], 4),
        'num_components_t0': network_features_t0['num_components'],
        'largest_component_ratio_t0': round(network_features_t0['largest_component_ratio'], 4),
        
        # Network features at t=T-1
        'density_tT': round(network_features_tT['density'], 4),
        'avg_degree_centrality_tT': round(network_features_tT['avg_degree_centrality'], 4),
        'max_degree_centrality_tT': round(network_features_tT['max_degree_centrality'], 4),
        'avg_betweenness_centrality_tT': round(network_features_tT['avg_betweenness_centrality'], 4) if network_features_tT['avg_betweenness_centrality'] is not None else None,
        'max_betweenness_centrality_tT': round(network_features_tT['max_betweenness_centrality'], 4) if network_features_tT['max_betweenness_centrality'] is not None else None,
        'avg_clustering_tT': round(network_features_tT['avg_clustering'], 4),
        'num_components_tT': network_features_tT['num_components'],
        'largest_component_ratio_tT': round(network_features_tT['largest_component_ratio'], 4),
        
        # Decision statistics (counts)
        'sell_t0': sell_t0,
        'vacant_t0': vacant_t0,
        'repair_t0': repair_t0,
        'sell_tT': sell_tT,
        'vacant_tT': vacant_tT,
        'repair_tT': repair_tT,
        
        # Decision statistics (percentages)
        'sell_pct_t0': round(100 * sell_t0 / total_households, 1),
        'vacant_pct_t0': round(100 * vacant_t0 / total_households, 1),
        'repair_pct_t0': round(100 * repair_t0 / total_households, 1),
        'sell_pct_tT': round(100 * sell_tT / total_households, 1),
        'vacant_pct_tT': round(100 * vacant_tT / total_households, 1),
        'repair_pct_tT': round(100 * repair_tT / total_households, 1)

        
    }

    states_tfinal = (
        house_states[house_states['time'] == house_states['time'].max()]
        [['repair_state', 'vacancy_state', 'sales_state']]
        .astype(int)
    )

    # 1) by number of decisions activated
    decisions_per_house = states_tfinal.sum(axis=1)
    dist = decisions_per_house.value_counts().sort_index()
    total_final = len(states_tfinal)

    print("\nBy number of decisions activated at final time (t = T-1):")
    for k in [0, 1, 2, 3]:
        cnt = int(dist.get(k, 0))
        pct = (100.0 * cnt / total_final) if total_final > 0 else 0.0
        print(f"  {k} decision(s): {cnt:5d} households ({pct:5.1f}%)")

    # 2) detailed combinations
    import numpy as np

    combo_counts = states_tfinal.value_counts().sort_index()

    # robustly convert MultiIndex/array keys to tuple(int,int,int)
    combo_dict = {
        tuple(map(int, np.atleast_1d(k))): int(v)
        for k, v in combo_counts.items()
    }

    print("\nDetailed combinations:")
    order = [
        (0,0,0), (1,0,0), (0,1,0), (0,0,1),
        (1,1,0), (1,0,1), (0,1,1), (1,1,1),
    ]
    labels = {
        (0,0,0): "None",
        (1,0,0): "Repair only",
        (0,1,0): "Vacant only",
        (0,0,1): "Sell only",
        (1,1,0): "Repair+Vacant",
        (1,0,1): "Repair+Sell",
        (0,1,1): "Vacant+Sell",
        (1,1,1): "All three",
    }
    for key in order:
        print(f"  {key} {labels[key]:15s}: {combo_dict.get(key, 0):5d}")
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """Print dataset statistics in a formatted way"""
    print(f"\n=== Dataset Statistics for {stats['scenario_name']} ===")
    print(f"Description: {stats['description']}")
    print(f"Blocking Mode: {stats.get('blocking_mode', 'N/A')}") 
    print(f"Total Households: {stats['total_households']}")
    print(f"Time Horizon: {stats['time_horizon']}")
    print()
    
    print("Network Statistics:")
    print(f"  Bonding links at t=0: {stats['bonding_links_t0']}")
    print(f"  Bridging links at t=0: {stats['bridging_links_t0']}")
    print(f"  Bonding links at t=T: {stats['bonding_links_tT']}")
    print(f"  Bridging links at t=T: {stats['bridging_links_tT']}")
    print(f"  Average degree at t=0: {stats['avg_degree_t0']}")
    print(f"  Average degree at t=T: {stats['avg_degree_tT']}")
    print()
    
    print("Network Features at t=0:")
    print(f"  Density: {stats['density_t0']}")
    print(f"  Avg Degree Centrality: {stats['avg_degree_centrality_t0']}")
    print(f"  Max Degree Centrality: {stats['max_degree_centrality_t0']}")
    print(f"  Avg Betweenness Centrality: {stats.get('avg_betweenness_centrality_t0', 'N/A')}")
    print(f"  Max Betweenness Centrality: {stats.get('max_betweenness_centrality_t0', 'N/A')}")
    print(f"  Avg Clustering: {stats['avg_clustering_t0']}")
    print(f"  Num Components: {stats['num_components_t0']}")
    print(f"  Largest Component Ratio: {stats['largest_component_ratio_t0']}")
    print()
    
    print("Decision Statistics:")
    print(f"  Sell at t=0: {stats['sell_t0']} ({stats['sell_pct_t0']}%)")
    print(f"  Vacant at t=0: {stats['vacant_t0']} ({stats['vacant_pct_t0']}%)")
    print(f"  Repair at t=0: {stats['repair_t0']} ({stats['repair_pct_t0']}%)")
    print(f"  Sell at t=T: {stats['sell_tT']} ({stats['sell_pct_tT']}%)")
    print(f"  Vacant at t=T: {stats['vacant_tT']} ({stats['vacant_pct_tT']}%)")
    print(f"  Repair at t=T: {stats['repair_tT']} ({stats['repair_pct_tT']}%)")


def save_statistics(stats: Dict[str, Any], output_folder: str):
    """Save statistics to JSON file"""
    with open(output_folder + '/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

def create_summary_excel(all_statistics: list, output_path: str):
    """Create Excel summary of all scenario statistics"""
    summary_df = pd.DataFrame(all_statistics)
    
    # Reorder columns for better readability
    base_columns = [
        'scenario_name', 'description', 'blocking_mode', 'total_households', 'time_horizon',
        'bonding_links_t0', 'bridging_links_t0', 'bonding_links_tT', 'bridging_links_tT',
        'avg_degree_t0', 'avg_degree_tT'
    ]
    
    network_features_columns = [
        'density_t0', 'avg_degree_centrality_t0', 'max_degree_centrality_t0',
        'avg_betweenness_centrality_t0', 'max_betweenness_centrality_t0',
        'avg_clustering_t0', 'num_components_t0', 'largest_component_ratio_t0',
        'density_tT', 'avg_degree_centrality_tT', 'max_degree_centrality_tT',
        'avg_betweenness_centrality_tT', 'max_betweenness_centrality_tT',
        'avg_clustering_tT', 'num_components_tT', 'largest_component_ratio_tT'
    ]
    
    decision_columns = [
        'sell_t0', 'vacant_t0', 'repair_t0', 'sell_tT', 'vacant_tT', 'repair_tT',
        'sell_pct_t0', 'vacant_pct_t0', 'repair_pct_t0', 'sell_pct_tT', 'vacant_pct_tT', 'repair_pct_tT'
    ]
    
    column_order = base_columns + network_features_columns + decision_columns
    
    summary_df = summary_df[column_order]
    summary_df.to_excel(output_path, index=False)
    print(f"\nSummary Excel saved to: {output_path}")
    
    return summary_df