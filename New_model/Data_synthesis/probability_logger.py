"""
probability_logger.py - Record detailed probability information from generator
"""
import numpy as np
import pandas as pd

def log_generator_probabilities(p_self_series, p_ji_df, links_df, house_states, t, k, 
                               house_df_with_features, geohash_to_model_idx):
    """
    Fixed version: use correct index mapping
    """
    detailed_entries = []
    homes = p_self_series.index.tolist()  # These are geohash strings
    
    # Use passed correct mapping
    home_to_idx = geohash_to_model_idx
    
    print(f"Debug: Generator logging at t={t}, k={k}")
    print(f"Debug: Using mapping with {len(home_to_idx)} entries")
    
    for home_i in homes:  # home_i is geohash
        # Check current state (house_states 'home' column is still geohash)
        current_state_mask = (house_states['time'] == t) & (house_states['home'] == home_i)
        if current_state_mask.sum() == 0:
            continue
            
        current_state = house_states[current_state_mask].iloc[0]
        state_cols = ['vacancy_state', 'repair_state', 'sales_state']
        if current_state[state_cols[k]] == 1:
            continue
        
        p_self_val = p_self_series.loc[home_i]
        
        # Find active and connected neighbors
        active_neighbors = []
        for home_j in homes:  # home_j is also geohash
            if home_i == home_j:
                continue
                
            # Check if neighbor is active
            neighbor_state_mask = (house_states['time'] == t) & (house_states['home'] == home_j)
            if neighbor_state_mask.sum() == 0:
                continue
                
            neighbor_state = house_states[neighbor_state_mask].iloc[0]
            if neighbor_state[state_cols[k]] == 1:
                # Check if there's a connection
                link_type = links_df.loc[home_i, home_j]  # links_df index is geohash
                if link_type > 0:
                    influence_prob = p_ji_df.loc[home_j, home_i]
                    
                    active_neighbors.append({
                        'neighbor_home': home_j,  # geohash
                        'neighbor_index': home_to_idx[home_j],  # 0-based model index
                        'link_type': int(link_type),
                        'influence_prob': float(influence_prob)
                    })
        
        # Calculate final probability
        if len(active_neighbors) == 0:
            final_activation_prob = p_self_val
            social_influence_term = 0.0
        else:
            neighbor_probs = [ni['influence_prob'] for ni in active_neighbors]
            product_term = np.prod([1 - p for p in neighbor_probs])
            social_influence_term = 1 - product_term
            final_activation_prob = 1 - (1 - p_self_val) * product_term
        
        detailed_entries.append({
            'timestep': int(t),
            'decision_type': int(k),
            'household_home': home_i,  # geohash
            'household_index': int(home_to_idx[home_i]),  # 0-based model index
            'self_activation_prob': float(p_self_val),
            'active_neighbors': len(active_neighbors),
            'neighbor_influences': active_neighbors,
            'social_influence_term': float(social_influence_term),
            'final_activation_prob': float(final_activation_prob),
            'generator_source': True
        })
    
    return detailed_entries