import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
from scipy.special import expit  # sigmoid function

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class TheoryAlignedDisasterRecoveryGenerator:
    def __init__(self, n_households=200, n_timesteps=25, observation_rate=0.3):
        """
        Generate synthetic data following the theoretical FR-SIC model
        
        Parameters:
        - n_households: Number of households in the community
        - n_timesteps: Number of time steps to simulate (0 to T)
        - observation_rate: Fraction of actual network links that are observed
        
        Constraint Handling:
        - Uses SOFT constraints through sigmoid probability modifications
        - Repair probability heavily penalized if vacant=1 or sell=1
        - Vacant probability encouraged if sell=1 (sell implies vacant)
        - Maintains differentiability for gradient-based learning
        - Avoids hard decision rules that would break the learning process
        """
        self.n_households = n_households
        self.n_timesteps = n_timesteps
        self.observation_rate = observation_rate
        
        # Generate household IDs
        self.household_ids = list(range(1, n_households + 1))
        
        # Initialize model parameters (following the theoretical formulation)
        self.init_model_parameters()
        
    def init_model_parameters(self):
        """Initialize parameters following the theoretical model"""
        
        # Network formation parameters
        self.alpha_0 = 0.3  # Reduced initial bonding formation strength
        
        # Observation model parameters  
        self.rho_1 = 0.3  # Missing rate for bonding links
        self.rho_2 = 0.6  # Missing rate for bridging links
        
        # Self-activation parameters (replacing NN with logistic regression)
        # Much lower intercepts and coefficients for gradual decision process
        self.self_params = {
            'vacant': {
                'intercept': -5.5,    # Much lower base probability
                'damage_coef': 1.2,   # Reduced damage influence
                'income_coef': -0.3,  # Higher income -> less likely to abandon
                'insurance_coef': -0.5,
                'time_coef': 0.03     # Slower time progression
            },
            'repair': {
                'intercept': -5.0,    # Much lower base probability
                'damage_coef': -0.3,  # More damage -> less likely to repair
                'income_coef': 0.8,   # Higher income -> more likely to repair
                'insurance_coef': 1.0,
                'time_coef': 0.02     # Slower time progression
            },
            'sell': {
                'intercept': -6.0,    # Lowest base probability
                'damage_coef': 1.0,   # Moderate damage influence
                'income_coef': 0.1,
                'insurance_coef': -0.2,
                'time_coef': 0.025    # Slower time progression
            }
        }
        
        # Influence parameters (replacing NN with logistic regression)
        # Reduced influence strength for more gradual diffusion
        self.influence_params = {
            'vacant': {
                'intercept': -3.5,    # Lower base influence
                'bonding_coef': 1.2,  # Reduced bonding influence
                'bridging_coef': 0.5, # Reduced bridging influence
                'distance_coef': -0.3,
                'similarity_coef': 0.4
            },
            'repair': {
                'intercept': -3.2,    # Lower base influence
                'bonding_coef': 1.4,  # Reduced bonding influence
                'bridging_coef': 0.6, # Reduced bridging influence
                'distance_coef': -0.2,
                'similarity_coef': 0.5
            },
            'sell': {
                'intercept': -4.0,    # Lower base influence
                'bonding_coef': 0.8,  # Reduced bonding influence
                'bridging_coef': 0.4, # Reduced bridging influence
                'distance_coef': -0.3,
                'similarity_coef': 0.3
            }
        }
        
        # Network formation parameters (replacing NN with logistic regression)
        # Reduced network formation for more realistic sparsity
        self.formation_params = {
            'intercept': -4.0,    # Much lower base formation probability
            'distance_coef': -1.5, # Stronger distance penalty
            'similarity_coef': 0.8, # Reduced similarity bonus
            'both_stayed_coef': 0.5, # Reduced staying bonus
            'time_coef': 0.01     # Very slow time progression
        }
        
    def generate_household_locations(self):
        """Generate household locations in a realistic community layout"""
        # Create clusters to simulate neighborhoods
        n_clusters = 5
        cluster_centers = [
            (40.7128 + np.random.normal(0, 0.01), -74.0060 + np.random.normal(0, 0.01)) 
            for _ in range(n_clusters)
        ]
        
        locations = []
        for hh_id in self.household_ids:
            # Assign household to a cluster
            cluster_idx = np.random.choice(n_clusters)
            center_lat, center_lon = cluster_centers[cluster_idx]
            
            # Add noise around cluster center
            lat = center_lat + np.random.normal(0, 0.005)
            lon = center_lon + np.random.normal(0, 0.005)
            
            locations.append({
                'household_id': hh_id,
                'latitude': lat,
                'longitude': lon
            })
        
        return pd.DataFrame(locations)
    
    def generate_household_features(self):
        """Generate household demographic and economic features"""
        features = []
        
        for hh_id in self.household_ids:
            # Generate correlated demographic features
            income_level = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            
            if income_level == 'low':
                income = np.random.normal(35000, 8000)
                property_value = np.random.normal(150000, 30000)
                insurance_coverage = np.random.choice([0, 1], p=[0.4, 0.6])
            elif income_level == 'medium':
                income = np.random.normal(65000, 15000)
                property_value = np.random.normal(280000, 50000)
                insurance_coverage = np.random.choice([0, 1], p=[0.2, 0.8])
            else:  # high
                income = np.random.normal(120000, 25000)
                property_value = np.random.normal(500000, 80000)
                insurance_coverage = np.random.choice([0, 1], p=[0.05, 0.95])
            
            features.append({
                'household_id': hh_id,
                'income': max(20000, income),  # Ensure minimum income
                'property_value': max(80000, property_value),  # Ensure minimum property value
                'insurance_coverage': insurance_coverage,
                'household_size': np.random.poisson(2.5) + 1,  # 1-6 people typically
                'age_head': np.random.normal(45, 15),
                'education_level': np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2]),
                'years_in_community': np.random.exponential(8),
                'damage_severity': np.random.uniform(0.1, 1.0)  # Proportion of home damaged
            })
        
        return pd.DataFrame(features)
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate approximate distance between two points in km"""
        lat_diff = (lat2 - lat1) * 111  # ~111 km per degree latitude
        lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def calculate_similarity(self, feat1, feat2):
        """Calculate demographic similarity between two households"""
        # Normalize features to [0,1] scale
        income_sim = 1 - abs(feat1['income'] - feat2['income']) / max(feat1['income'], feat2['income'])
        age_sim = 1 - abs(feat1['age_head'] - feat2['age_head']) / 50
        edu_sim = 1 - abs(feat1['education_level'] - feat2['education_level']) / 3
        
        # Weight average
        similarity = 0.4 * income_sim + 0.3 * age_sim + 0.3 * edu_sim
        return max(0, min(1, similarity))
    
    def calculate_interaction_potential(self, feat1, feat2, distance, similarity, timestep):
        """Calculate interaction potential using logistic regression (replacing NN_form)"""
        # Check if both households stayed (haven't relocated)
        both_stayed = (feat1.get('vacant', 0) == 0) and (feat2.get('vacant', 0) == 0)
        
        linear_combination = (
            self.formation_params['intercept'] +
            self.formation_params['distance_coef'] * distance +
            self.formation_params['similarity_coef'] * similarity +
            self.formation_params['both_stayed_coef'] * float(both_stayed) +
            self.formation_params['time_coef'] * timestep
        )
        
        return expit(linear_combination)  # sigmoid function
    
    def generate_initial_network(self, locations_df, features_df):
        """Generate initial network following the prior distribution p(ℓ_ij(0))"""
        network_links = []
        
        for i, hh1 in enumerate(self.household_ids):
            for j, hh2 in enumerate(self.household_ids[i+1:], i+1):
                
                # Get locations and features
                loc1 = locations_df[locations_df['household_id'] == hh1].iloc[0]
                loc2 = locations_df[locations_df['household_id'] == hh2].iloc[0]
                feat1 = features_df[features_df['household_id'] == hh1].iloc[0]
                feat2 = features_df[features_df['household_id'] == hh2].iloc[0]
                
                distance = self.calculate_distance(
                    loc1['latitude'], loc1['longitude'],
                    loc2['latitude'], loc2['longitude']
                )
                
                similarity = self.calculate_similarity(feat1, feat2)
                
                # Calculate interaction potential following theory
                interaction_potential = self.calculate_interaction_potential(
                    feat1, feat2, distance, similarity, 0
                )
                
                # Calculate initial link probabilities following theory
                # logits_0 = [0, log(α_0 × similarity + ε), log(interaction_potential + ε)]
                epsilon = 1e-6
                logits = np.array([
                    0,  # No link
                    np.log(self.alpha_0 * similarity + epsilon),  # Bonding
                    np.log(interaction_potential + epsilon)  # Bridging
                ])
                
                # Apply softmax to get probabilities
                exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
                probs = exp_logits / np.sum(exp_logits)
                
                # Sample link type
                link_type = np.random.choice([0, 1, 2], p=probs)
                
                if link_type > 0:
                    network_links.append({
                        'household_id_1': hh1,
                        'household_id_2': hh2,
                        'link_type': link_type,
                        'timestep': 0
                    })
        
        return network_links
    
    def calculate_self_activation_prob(self, household_features, decision_type, timestep, current_state):
        """Calculate self-activation probability using logistic regression with soft constraints"""
        params = self.self_params[decision_type]
        
        # Normalize income to reasonable scale
        income_normalized = household_features['income'] / 100000
        
        # Base linear combination
        linear_combination = (
            params['intercept'] +
            params['damage_coef'] * household_features['damage_severity'] +
            params['income_coef'] * income_normalized +
            params['insurance_coef'] * household_features['insurance_coverage'] +
            params['time_coef'] * timestep
        )
        
        # Apply soft constraints through probability modifications
        if decision_type == 'repair':
            # Strong penalty for repair if already vacant or sold
            if current_state['vacant'] == 1:
                linear_combination -= 10.0  # Very strong penalty (but not -∞)
            if current_state['sell'] == 1:
                linear_combination -= 10.0  # Very strong penalty (but not -∞)
                
        elif decision_type == 'vacant':
            # Strong bonus for vacant if already sold (sell implies vacant)
            if current_state['sell'] == 1:
                linear_combination += 8.0  # Strong encouragement to become vacant
                
        elif decision_type == 'sell':
            # Moderate penalty for sell if already vacant (harder to sell abandoned property)
            if current_state['vacant'] == 1:
                linear_combination -= 2.0  # Moderate penalty
        
        return expit(linear_combination)
    
    def calculate_influence_prob(self, influencer_id, influenced_id, link_type, decision_type, 
                               features_df, locations_df, timestep):
        """Calculate influence probability using logistic regression (replacing NN_influence)"""
        params = self.influence_params[decision_type]
        
        # Get features
        feat1 = features_df[features_df['household_id'] == influencer_id].iloc[0]
        feat2 = features_df[features_df['household_id'] == influenced_id].iloc[0]
        loc1 = locations_df[locations_df['household_id'] == influencer_id].iloc[0]
        loc2 = locations_df[locations_df['household_id'] == influenced_id].iloc[0]
        
        distance = self.calculate_distance(
            loc1['latitude'], loc1['longitude'],
            loc2['latitude'], loc2['longitude']
        )
        similarity = self.calculate_similarity(feat1, feat2)
        
        # Create binary indicators for link types
        is_bonding = float(link_type == 1)
        is_bridging = float(link_type == 2)
        
        linear_combination = (
            params['intercept'] +
            params['bonding_coef'] * is_bonding +
            params['bridging_coef'] * is_bridging +
            params['distance_coef'] * distance +
            params['similarity_coef'] * similarity
        )
        
        return expit(linear_combination)
    
    def simulate_fr_sic_process(self, features_df, locations_df, initial_network):
        """Simulate the First-Round Sequential Independent Cascade process"""
        states = []
        network_evolution = []
        
        # Initialize states - all households start undecided
        current_states = {hh_id: {'vacant': 0, 'repair': 0, 'sell': 0} for hh_id in self.household_ids}
        
        # Add initial states (t=0)
        for hh_id in self.household_ids:
            states.append({
                'household_id': hh_id,
                'timestep': 0,
                'vacant': 0,
                'repair': 0,
                'sell': 0
            })
        
        # Initialize network
        current_network = {(link['household_id_1'], link['household_id_2']): link['link_type'] 
                          for link in initial_network}
        network_evolution.extend(initial_network)
        
        # Simulate evolution for each timestep
        for t in range(1, self.n_timesteps):
            new_states = {}
            
            # For each household, apply FR-SIC process
            for hh_id in self.household_ids:
                new_states[hh_id] = current_states[hh_id].copy()
                
                # For each decision type
                for decision_type in ['vacant', 'repair', 'sell']:
                    # Skip if already decided (irreversibility constraint)
                    if current_states[hh_id][decision_type] == 1:
                        continue
                    
                    # Calculate self-activation probability with current state for constraints
                    hh_features = features_df[features_df['household_id'] == hh_id].iloc[0]
                    p_self = self.calculate_self_activation_prob(
                        hh_features, decision_type, t, current_states[hh_id]
                    )
                    
                    # Calculate social influence
                    # Find active neighbors for this decision type
                    active_neighbors = {}  # {link_type: [neighbor_ids]}
                    for link_type in [1, 2]:  # bonding, bridging
                        active_neighbors[link_type] = []
                        
                        for (hh1, hh2), net_link_type in current_network.items():
                            if net_link_type == link_type:
                                neighbor_id = None
                                if hh1 == hh_id and current_states[hh2][decision_type] == 1:
                                    neighbor_id = hh2
                                elif hh2 == hh_id and current_states[hh1][decision_type] == 1:
                                    neighbor_id = hh1
                                
                                if neighbor_id:
                                    active_neighbors[link_type].append(neighbor_id)
                    
                    # Apply FR-SIC formula: P = 1 - (1 - p_self) * Π(1 - p_influence)
                    prob_not_activated = (1 - p_self)
                    
                    for link_type in [1, 2]:
                        for neighbor_id in active_neighbors[link_type]:
                            p_influence = self.calculate_influence_prob(
                                neighbor_id, hh_id, link_type, decision_type,
                                features_df, locations_df, t
                            )
                            prob_not_activated *= (1 - p_influence)
                    
                    activation_prob = 1 - prob_not_activated
                    
                    # Make decision
                    if np.random.random() < activation_prob:
                        new_states[hh_id][decision_type] = 1
            
            # Update states
            current_states = new_states
            
            # Record states
            for hh_id in self.household_ids:
                states.append({
                    'household_id': hh_id,
                    'timestep': t,
                    'vacant': current_states[hh_id]['vacant'],
                    'repair': current_states[hh_id]['repair'],
                    'sell': current_states[hh_id]['sell']
                })
            
            # Evolve network following theoretical transition model
            new_network = self.evolve_network(current_network, current_states, features_df, 
                                            locations_df, t)
            current_network = new_network
            
            # Record network evolution
            for (hh1, hh2), link_type in current_network.items():
                network_evolution.append({
                    'household_id_1': hh1,
                    'household_id_2': hh2,
                    'link_type': link_type,
                    'timestep': t
                })
        
        return pd.DataFrame(states), pd.DataFrame(network_evolution)
    
    def evolve_network(self, prev_network, current_states, features_df, locations_df, timestep):
        """Evolve network following theoretical transition probabilities"""
        new_network = {}
        
        # All possible pairs
        all_pairs = set()
        for i, hh1 in enumerate(self.household_ids):
            for hh2 in self.household_ids[i+1:]:
                all_pairs.add((hh1, hh2))
        
        for (hh1, hh2) in all_pairs:
            prev_link_type = prev_network.get((hh1, hh2), 0)
            
            # Get current states
            state1 = current_states[hh1]
            state2 = current_states[hh2]
            
            # Calculate transition probabilities
            if prev_link_type == 0:  # No previous connection
                # Can only form bridging links, and only if both haven't relocated
                if state1['vacant'] == 1 or state2['vacant'] == 1:
                    new_link_type = 0  # Cannot form new connections if relocated
                else:
                    # Calculate interaction potential
                    feat1 = features_df[features_df['household_id'] == hh1].iloc[0]
                    feat2 = features_df[features_df['household_id'] == hh2].iloc[0]
                    loc1 = locations_df[locations_df['household_id'] == hh1].iloc[0]
                    loc2 = locations_df[locations_df['household_id'] == hh2].iloc[0]
                    
                    distance = self.calculate_distance(
                        loc1['latitude'], loc1['longitude'],
                        loc2['latitude'], loc2['longitude']
                    )
                    similarity = self.calculate_similarity(feat1, feat2)
                    
                    # Add current states to features for interaction potential
                    feat1_with_state = dict(feat1)
                    feat2_with_state = dict(feat2)
                    feat1_with_state.update(state1)
                    feat2_with_state.update(state2)
                    
                    interaction_potential = self.calculate_interaction_potential(
                        feat1_with_state, feat2_with_state, distance, similarity, timestep
                    )
                    
                    # Transition probabilities from no connection
                    if np.random.random() < interaction_potential:
                        new_link_type = 2  # Form bridging link
                    else:
                        new_link_type = 0  # Stay disconnected
            
            elif prev_link_type == 1:  # Previous bonding link
                # Bonding links persist (assumption: p_11 = 1)
                new_link_type = 1
            
            elif prev_link_type == 2:  # Previous bridging link
                # Bridging links may decay
                if state1['vacant'] == 1 or state2['vacant'] == 1:
                    # Calculate decay probability when someone relocated
                    feat1 = features_df[features_df['household_id'] == hh1].iloc[0]
                    feat2 = features_df[features_df['household_id'] == hh2].iloc[0]
                    loc1 = locations_df[locations_df['household_id'] == hh1].iloc[0]
                    loc2 = locations_df[locations_df['household_id'] == hh2].iloc[0]
                    
                    distance = self.calculate_distance(
                        loc1['latitude'], loc1['longitude'],
                        loc2['latitude'], loc2['longitude']
                    )
                    similarity = self.calculate_similarity(feat1, feat2)
                    
                    # Add current states to features
                    feat1_with_state = dict(feat1)
                    feat2_with_state = dict(feat2)
                    feat1_with_state.update(state1)
                    feat2_with_state.update(state2)
                    
                    interaction_potential = self.calculate_interaction_potential(
                        feat1_with_state, feat2_with_state, distance, similarity, timestep
                    )
                    
                    # p_22 = interaction_potential, p_20 = 1 - interaction_potential
                    if np.random.random() < interaction_potential:
                        new_link_type = 2  # Maintain bridging link
                    else:
                        new_link_type = 0  # Link disappears
                else:
                    # Both stayed: bridging link persists (p_22 = 1)
                    new_link_type = 2
            
            # Add to new network if link exists
            if new_link_type > 0:
                new_network[(hh1, hh2)] = new_link_type
        
        return new_network
    
    def apply_observation_model(self, complete_network_df):
        """Apply observation model following p(ℓ_ij^obs(t) | ℓ_ij(t), θ_obs)"""
        observed_networks = []
        
        for _, link in complete_network_df.iterrows():
            link_type = link['link_type']
            
            # Get missing rate based on link type
            if link_type == 1:  # Bonding
                missing_rate = self.rho_1
            elif link_type == 2:  # Bridging
                missing_rate = self.rho_2
            else:
                continue  # No link (type 0) is always missing
            
            # Observation probability = 1 - missing_rate
            observation_prob = 1 - missing_rate
            
            if np.random.random() < observation_prob:
                observed_networks.append({
                    'household_id_1': link['household_id_1'],
                    'household_id_2': link['household_id_2'],
                    'link_type': link['link_type'],
                    'timestep': link['timestep']
                })
        
        return pd.DataFrame(observed_networks)
    
    def print_summary(self, states_df, complete_network_df, observed_network_df, initial_network):
        """Print detailed summary following theoretical model"""
        print("\n=== Theory-Aligned Data Generation Summary ===")
        print(f"Model: First-Round Sequential Independent Cascade (FR-SIC)")
        print(f"Number of households: {self.n_households}")
        print(f"Number of timesteps: {self.n_timesteps}")
        print(f"Observation rates: ρ₁={self.rho_1:.1%} (bonding), ρ₂={self.rho_2:.1%} (bridging)")
        
        # Network statistics
        total_gt_edges = len(complete_network_df)
        total_obs_edges = len(observed_network_df)
        
        print(f"\nNetwork Evolution:")
        print(f"Total ground truth network entries: {total_gt_edges}")
        print(f"Total observed network entries: {total_obs_edges}")
        
        # By link type
        gt_bonding = len(complete_network_df[complete_network_df['link_type'] == 1])
        gt_bridging = len(complete_network_df[complete_network_df['link_type'] == 2])
        obs_bonding = len(observed_network_df[observed_network_df['link_type'] == 1])
        obs_bridging = len(observed_network_df[observed_network_df['link_type'] == 2])
        
        print(f"Ground truth - Bonding: {gt_bonding}, Bridging: {gt_bridging}")
        print(f"Observed - Bonding: {obs_bonding}, Bridging: {obs_bridging}")
        
        if gt_bonding > 0:
            actual_bonding_obs_rate = obs_bonding / gt_bonding
            print(f"Actual bonding observation rate: {actual_bonding_obs_rate:.1%} (target: {1-self.rho_1:.1%})")
        if gt_bridging > 0:
            actual_bridging_obs_rate = obs_bridging / gt_bridging
            print(f"Actual bridging observation rate: {actual_bridging_obs_rate:.1%} (target: {1-self.rho_2:.1%})")
        
        # Decision dynamics
        print(f"\nDecision Dynamics (FR-SIC Process):")
        final_states = states_df[states_df['timestep'] == self.n_timesteps - 1]
        
        n_vacant = final_states['vacant'].sum()
        n_repair = final_states['repair'].sum()
        n_sell = final_states['sell'].sum()
        n_undecided = len(final_states) - n_vacant - n_repair - n_sell
        
        print(f"Final decisions:")
        print(f"  Vacant (relocated): {n_vacant} ({n_vacant/len(final_states):.1%})")
        print(f"  Repair: {n_repair} ({n_repair/len(final_states):.1%})")
        print(f"  Sell: {n_sell} ({n_sell/len(final_states):.1%})")
        print(f"  Undecided: {n_undecided} ({n_undecided/len(final_states):.1%})")
        
        # Decision timing analysis
        print(f"\nDecision Timing Analysis:")
        for decision_type in ['vacant', 'repair', 'sell']:
            decision_times = []
            for hh_id in self.household_ids:
                hh_states = states_df[states_df['household_id'] == hh_id].sort_values('timestep')
                decision_made = hh_states[hh_states[decision_type] == 1]
                if len(decision_made) > 0:
                    first_decision_time = decision_made.iloc[0]['timestep']
                    decision_times.append(first_decision_time)
            
            if decision_times:
                avg_time = np.mean(decision_times)
                print(f"  {decision_type.capitalize()}: {len(decision_times)} decisions, avg time = {avg_time:.1f}")
        
        # Network evolution over time
        print(f"\nNetwork Evolution Over Time:")
        initial_links = len([l for l in initial_network if l])
        print(f"Initial network links: {initial_links}")
        
        for t in [5, 10, 15, 20, self.n_timesteps-1]:
            if t < self.n_timesteps:
                links_at_t = len(complete_network_df[complete_network_df['timestep'] == t])
                bonding_at_t = len(complete_network_df[
                    (complete_network_df['timestep'] == t) & 
                    (complete_network_df['link_type'] == 1)
                ])
                bridging_at_t = len(complete_network_df[
                    (complete_network_df['timestep'] == t) & 
                    (complete_network_df['link_type'] == 2)
                ])
                print(f"  t={t}: {links_at_t} total ({bonding_at_t} bonding, {bridging_at_t} bridging)")
    
    def generate_all_data(self, output_dir='data'):
        """Generate all data files following theoretical model"""
        print("Generating theory-aligned post-disaster recovery data...")
        print(f"Following FR-SIC process with {self.n_households} households over {self.n_timesteps} timesteps")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate location data
        print("1. Generating household locations...")
        locations_df = self.generate_household_locations()
        locations_df.to_csv(f'{output_dir}/household_locations.csv', index=False)
        
        # Generate household features
        print("2. Generating household features...")
        features_df = self.generate_household_features()
        features_df.to_csv(f'{output_dir}/household_features.csv', index=False)
        
        # Generate initial network following prior distribution
        print("3. Generating initial network following prior p(ℓ_ij(0))...")
        initial_network = self.generate_initial_network(locations_df, features_df)
        
        # Simulate FR-SIC process and network co-evolution
        print("4. Simulating FR-SIC decision process and network co-evolution...")
        states_df, complete_network_df = self.simulate_fr_sic_process(
            features_df, locations_df, initial_network
        )
        states_df.to_csv(f'{output_dir}/household_states.csv', index=False)
        complete_network_df.to_csv(f'{output_dir}/ground_truth_network.csv', index=False)
        
        # Apply observation model
        print("5. Applying observation model p(G^obs | G, θ_obs)...")
        observed_network_df = self.apply_observation_model(complete_network_df)
        observed_network_df.to_csv(f'{output_dir}/observed_network.csv', index=False)
        
        # Generate detailed summary
        self.print_summary(states_df, complete_network_df, observed_network_df, initial_network)
        
        print(f"\nData files saved to '{output_dir}/' directory:")
        print("- household_locations.csv")
        print("- household_features.csv") 
        print("- household_states.csv")
        print("- ground_truth_network.csv (complete network)")
        print("- observed_network.csv (partial observations)")
        
        return {
            'locations': locations_df,
            'features': features_df,
            'states': states_df,
            'ground_truth_network': complete_network_df,
            'observed_network': observed_network_df
        }


# Generate the data
if __name__ == "__main__":
    # Set working directory - UPDATE THIS PATH TO YOUR DESIRED LOCATION
    os.chdir('/Users/susangao/Desktop/CLIMA/CODE 4/data/syn_data_v3') 
    
    # Create generator with theoretical parameters
    generator = TheoryAlignedDisasterRecoveryGenerator(
        n_households=49,      # Medium-sized community
        n_timesteps=25,        # ~2 years with monthly observations
        observation_rate=0.3   # Not directly used, replaced by ρ₁, ρ₂
    )
    
    # Generate all data following theoretical model
    data = generator.generate_all_data(output_dir='theory_aligned_data')
    
    print("\n=== Sample Data Preview ===")
    print("\nHousehold Locations (first 5 rows):")
    print(data['locations'].head())
    
    print("\nHousehold Features (first 3 rows):")
    print(data['features'].head(3))
    
    print("\nHousehold States (first 10 rows):")
    print(data['states'].head(10))
    
    print("\nGround Truth Network (first 10 rows):")
    print(data['ground_truth_network'].head(10))
    
    print("\nObserved Network (first 10 rows):")
    print(data['observed_network'].head(10))
    
    # Additional analysis for verification
    print("\n=== Model Verification ===")
    
    # Verify FR-SIC properties
    states_df = data['states']
    print("\nFR-SIC Property Verification:")
    
    # Check irreversibility
    violations = 0
    for hh_id in generator.household_ids:
        hh_states = states_df[states_df['household_id'] == hh_id].sort_values('timestep')
        for decision_type in ['vacant', 'repair', 'sell']:
            decisions = hh_states[decision_type].values
            for i in range(1, len(decisions)):
                if decisions[i] < decisions[i-1]:  # Decision reversed
                    violations += 1
    
    print(f"Irreversibility violations: {violations} (should be 0)")
    
    # Check logical constraints (should now have fewer violations due to soft constraints)
    constraint_violations = 0
    total_constraint_checks = 0
    final_states = states_df[states_df['timestep'] == generator.n_timesteps - 1]
    for _, row in final_states.iterrows():
        total_constraint_checks += 1
        # Can't repair if vacant or sold (should be rare due to soft constraints)
        if row['repair'] == 1 and (row['vacant'] == 1 or row['sell'] == 1):
            constraint_violations += 1
    
    print(f"Logical constraint violations: {constraint_violations}/{total_constraint_checks} ({constraint_violations/total_constraint_checks:.1%})")
    
    # Check sell->vacant consistency
    sell_vacant_consistency = 0
    sell_households = final_states[final_states['sell'] == 1]
    if len(sell_households) > 0:
        sell_and_vacant = len(sell_households[sell_households['vacant'] == 1])
        sell_vacant_consistency = sell_and_vacant / len(sell_households)
    print(f"Sell->Vacant consistency: {sell_vacant_consistency:.1%} (higher is better)")
    
    # Verify network evolution properties
    network_df = data['ground_truth_network']
    print(f"\nNetwork Evolution Property Verification:")
    
    # Check bonding persistence
    bonding_violations = 0
    for hh1 in generator.household_ids:
        for hh2 in generator.household_ids:
            if hh1 >= hh2:
                continue
            
            pair_links = network_df[
                ((network_df['household_id_1'] == hh1) & (network_df['household_id_2'] == hh2)) |
                ((network_df['household_id_1'] == hh2) & (network_df['household_id_2'] == hh1))
            ].sort_values('timestep')
            
            if len(pair_links) > 0:
                bonding_found = False
                for _, link in pair_links.iterrows():
                    if link['link_type'] == 1:
                        if bonding_found:
                            continue  # Should persist
                        bonding_found = True
                    elif bonding_found and link['link_type'] != 1:
                        bonding_violations += 1  # Bonding link disappeared
    
    print(f"Bonding persistence violations: {bonding_violations} (should be 0)")
    
    print(f"\n=== Data Generation Complete ===")
    print(f"Theory-aligned data successfully generated following:")
    print(f"- FR-SIC decision diffusion process")
    print(f"- Network evolution with co-evolution dynamics") 
    print(f"- Proper observation model with link-type specific missing rates")
    print(f"- All theoretical constraints and assumptions")
    print(f"- Soft constraint penalties for logical consistency")