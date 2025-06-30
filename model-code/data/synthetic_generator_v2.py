import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(40)
random.seed(40)

class DisasterRecoveryDataGeneratorV2:
    def __init__(self, n_households=200, n_timesteps=25, observation_rate=0.3):
        """
        Generate synthetic data for post-disaster recovery network analysis (Version 2)
        
        Key improvements:
        1. Stronger feature-link relationships
        2. More realistic community structure
        3. Clearer temporal dynamics
        4. Better balance between link types
        """
        self.n_households = n_households
        self.n_timesteps = n_timesteps
        self.observation_rate = observation_rate
        
        # Generate household IDs
        self.household_ids = list(range(1, n_households + 1))
        
    def generate_household_locations(self):
        """Generate household locations with realistic clustering."""
        # Create 3 main neighborhoods with clear geographical separation
        neighborhoods = [
            {'center': (40.7128, -74.0060), 'households': self.n_households // 3},      # Downtown
            {'center': (40.7489, -73.9857), 'households': self.n_households // 3},      # Midtown  
            {'center': (40.7831, -73.9712), 'households': self.n_households - 2*(self.n_households // 3)}  # Uptown
        ]
        
        locations = []
        hh_idx = 0
        
        for neighborhood in neighborhoods:
            center_lat, center_lon = neighborhood['center']
            n_hh = neighborhood['households']
            
            for _ in range(n_hh):
                if hh_idx >= len(self.household_ids):
                    break
                    
                # Cluster around neighborhood center with some spread
                lat = center_lat + np.random.normal(0, 0.008)  # ~800m spread
                lon = center_lon + np.random.normal(0, 0.008)
                
                locations.append({
                    'household_id': self.household_ids[hh_idx],
                    'latitude': lat,
                    'longitude': lon
                })
                hh_idx += 1
        
        return pd.DataFrame(locations)
    
    def generate_household_features(self):
        """Generate household features with clear socioeconomic clustering."""
        features = []
        
        # Create 3 socioeconomic groups that align with neighborhoods
        n_per_group = self.n_households // 3
        
        for group_idx in range(3):
            start_idx = group_idx * n_per_group
            end_idx = start_idx + n_per_group if group_idx < 2 else len(self.household_ids)
            
            if group_idx == 0:  # Lower income group
                income_base, income_std = 40000, 8000
                prop_value_base, prop_value_std = 180000, 40000
                insurance_rate = 0.4
                education_weights = [0.5, 0.3, 0.15, 0.05]  # Lower education
            elif group_idx == 1:  # Middle income group  
                income_base, income_std = 70000, 12000
                prop_value_base, prop_value_std = 320000, 60000
                insurance_rate = 0.8
                education_weights = [0.2, 0.4, 0.3, 0.1]   # Mixed education
            else:  # Higher income group
                income_base, income_std = 110000, 20000
                prop_value_base, prop_value_std = 520000, 80000
                insurance_rate = 0.95
                education_weights = [0.05, 0.2, 0.4, 0.35]  # Higher education
            
            for i in range(start_idx, end_idx):
                if i >= len(self.household_ids):
                    break
                    
                hh_id = self.household_ids[i]
                
                # Generate correlated features within group
                income = max(25000, np.random.normal(income_base, income_std))
                property_value = max(100000, np.random.normal(prop_value_base, prop_value_std))
                insurance_coverage = np.random.choice([0, 1], p=[1-insurance_rate, insurance_rate])
                
                # Age correlates somewhat with income in each group
                age_base = 35 + (income - income_base) / income_std * 5
                age_head = max(25, min(75, np.random.normal(age_base, 12)))
                
                features.append({
                    'household_id': hh_id,
                    'income': income,
                    'property_value': property_value,
                    'insurance_coverage': insurance_coverage,
                    'household_size': np.random.poisson(2.2) + 1,  # 1-5 people typically
                    'age_head': age_head,
                    'education_level': np.random.choice([1, 2, 3, 4], p=education_weights),
                    'years_in_community': np.random.exponential(7) + 1,
                    'damage_severity': np.random.uniform(0.2, 0.9)  # Significant damage for all
                })
        
        return pd.DataFrame(features)
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate approximate distance between two points in km."""
        lat_diff = (lat2 - lat1) * 111  # ~111 km per degree latitude
        lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def calculate_feature_similarity(self, feat1, feat2):
        """Calculate comprehensive similarity between two households."""
        # Normalize features to 0-1 scale for comparison
        income_sim = 1 - abs(feat1['income'] - feat2['income']) / max(feat1['income'], feat2['income'])
        age_sim = 1 - abs(feat1['age_head'] - feat2['age_head']) / 50
        edu_sim = 1 - abs(feat1['education_level'] - feat2['education_level']) / 3
        size_sim = 1 - abs(feat1['household_size'] - feat2['household_size']) / 4
        damage_sim = 1 - abs(feat1['damage_severity'] - feat2['damage_severity'])
        
        # Weighted average (income and education matter more for bonding)
        overall_similarity = (0.3 * income_sim + 0.2 * age_sim + 0.3 * edu_sim + 
                            0.1 * size_sim + 0.1 * damage_sim)
        
        return overall_similarity
    
    def generate_initial_network(self, locations_df, features_df):
        """Generate initial network with strong feature-distance relationships."""
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
                
                similarity = self.calculate_feature_similarity(feat1, feat2)
                
                # BONDING links: High similarity regardless of distance (family, close friends)
                # Make this much stronger and more predictable
                bonding_prob = 0.12 * (similarity ** 2)  # 0-12% based on similarity
                if similarity > 0.8:  # Very similar households
                    bonding_prob *= 2.5  # Boost to 30% for very similar
                bonding_prob = min(bonding_prob, 0.35)
                
                # BRIDGING links: Proximity-based with some similarity boost
                # Make spatial relationships much clearer
                if distance < 0.8:  # Within 800m - same neighborhood
                    bridging_prob = 0.45 * (1 - distance/0.8) * (0.6 + 0.4 * similarity)
                elif distance < 2.5:  # Within 2.5km - nearby neighborhoods
                    bridging_prob = 0.25 * (1 - (distance-0.8)/(2.5-0.8)) * similarity
                else:  # Distant
                    bridging_prob = 0.03 * similarity
                
                # Ensure mutual exclusivity and reasonable total probability
                total_link_prob = bonding_prob + bridging_prob
                if total_link_prob > 0.7:
                    # Rescale to prevent too many links
                    bonding_prob *= 0.7 / total_link_prob
                    bridging_prob *= 0.7 / total_link_prob
                
                # Decision
                rand_val = np.random.random()
                if rand_val < bonding_prob:
                    link_type = 1  # Bonding
                elif rand_val < bonding_prob + bridging_prob:
                    link_type = 2  # Bridging
                else:
                    link_type = 0  # No link
                
                if link_type > 0:
                    network_links.append({
                        'household_id_1': hh1,
                        'household_id_2': hh2,
                        'link_type': link_type,
                        'timestep': 0
                    })
        
        return network_links
    
    def simulate_state_evolution(self, features_df, initial_network):
        """Simulate household decision evolution with stronger network effects."""
        states = []
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
        
        # Create network lookup
        current_network = {(link['household_id_1'], link['household_id_2']): link['link_type'] 
                         for link in initial_network}
        
        for t in range(1, self.n_timesteps):
            for hh_id in self.household_ids:
                # Skip if already decided
                if (current_states[hh_id]['vacant'] == 1 or 
                    current_states[hh_id]['repair'] == 1 or 
                    current_states[hh_id]['sell'] == 1):
                    states.append({
                        'household_id': hh_id,
                        'timestep': t,
                        'vacant': current_states[hh_id]['vacant'],
                        'repair': current_states[hh_id]['repair'],
                        'sell': current_states[hh_id]['sell']
                    })
                    continue
                
                # Get household features
                hh_features = features_df[features_df['household_id'] == hh_id].iloc[0]
                
                # Calculate BASE probabilities from individual characteristics
                damage = hh_features['damage_severity']
                income_norm = hh_features['income'] / 100000  # Normalize to ~0-1
                insurance = hh_features['insurance_coverage']
                
                # Stronger individual effects
                base_vacant_prob = 0.015 + 0.08 * damage * (1.5 - insurance) * (1.2 - income_norm)
                base_repair_prob = 0.02 + 0.12 * insurance * income_norm * (0.8 + 0.2 * damage)
                base_sell_prob = 0.01 + 0.05 * damage * (1 - insurance)
                
                # Add STRONG social influence
                connected_neighbors = []
                for (hh1, hh2), link_type in current_network.items():
                    if hh1 == hh_id:
                        connected_neighbors.append((hh2, link_type))
                    elif hh2 == hh_id:
                        connected_neighbors.append((hh1, link_type))
                
                # Calculate social influence with clear differentiation
                influence_vacant = 0
                influence_repair = 0
                influence_sell = 0
                
                for neighbor_id, link_type in connected_neighbors:
                    neighbor_state = current_states[neighbor_id]
                    
                    # Much stronger and differentiated influence
                    if link_type == 1:  # Bonding - strong influence
                        bonding_strength = 0.8
                    else:  # Bridging - moderate influence  
                        bridging_strength = 0.3
                    
                    if neighbor_state['vacant'] == 1:
                        influence_vacant += bonding_strength * 0.6 if link_type == 1 else bridging_strength * 0.4
                    if neighbor_state['repair'] == 1:
                        influence_repair += bonding_strength * 0.7 if link_type == 1 else bridging_strength * 0.5
                    if neighbor_state['sell'] == 1:
                        influence_sell += bonding_strength * 0.5 if link_type == 1 else bridging_strength * 0.3
                
                # Final probabilities with clear social amplification
                vacant_prob = min(0.8, base_vacant_prob + influence_vacant)
                repair_prob = min(0.8, base_repair_prob + influence_repair)
                sell_prob = min(0.8, base_sell_prob + influence_sell)
                
                # Normalize to ensure reasonable total probability
                total_prob = vacant_prob + repair_prob + sell_prob
                if total_prob > 0.9:
                    vacant_prob *= 0.9 / total_prob
                    repair_prob *= 0.9 / total_prob
                    sell_prob *= 0.9 / total_prob
                
                # Make decision
                rand_val = np.random.random()
                if rand_val < vacant_prob:
                    current_states[hh_id]['vacant'] = 1
                elif rand_val < vacant_prob + repair_prob:
                    current_states[hh_id]['repair'] = 1
                elif rand_val < vacant_prob + repair_prob + sell_prob:
                    current_states[hh_id]['sell'] = 1
                
                # Record state
                states.append({
                    'household_id': hh_id,
                    'timestep': t,
                    'vacant': current_states[hh_id]['vacant'],
                    'repair': current_states[hh_id]['repair'],
                    'sell': current_states[hh_id]['sell']
                })
        
        return pd.DataFrame(states)
    
    def simulate_network_evolution_complete(self, initial_network, states_df):
        """Simulate network evolution with clearer temporal patterns."""
        all_networks = []
        
        # Add initial network
        all_networks.extend(initial_network)
        
        for t in range(1, self.n_timesteps):
            current_network = []
            
            # Get previous network
            prev_network = {(link['household_id_1'], link['household_id_2']): link['link_type'] 
                           for link in all_networks if link['timestep'] == t-1}
            
            for (hh1, hh2), prev_link_type in prev_network.items():
                # Get current states
                state1 = states_df[(states_df['household_id'] == hh1) & (states_df['timestep'] == t)].iloc[0]
                state2 = states_df[(states_df['household_id'] == hh2) & (states_df['timestep'] == t)].iloc[0]
                
                # BONDING links: Nearly always persist (strong relationships)
                if prev_link_type == 1:
                    persist_prob = 0.98  # Very high persistence
                    if state1['vacant'] == 1 or state2['vacant'] == 1:
                        persist_prob = 0.85  # Slight reduction if someone moved
                    
                    if np.random.random() < persist_prob:
                        current_network.append({
                            'household_id_1': hh1,
                            'household_id_2': hh2,
                            'link_type': 1,
                            'timestep': t
                        })
                
                # BRIDGING links: More variable persistence
                elif prev_link_type == 2:
                    base_persist = 0.8  # Good baseline persistence
                    
                    # Strong decay if someone moved out
                    if state1['vacant'] == 1 or state2['vacant'] == 1:
                        persist_prob = 0.2  # Sharp drop
                    # Slight boost if both repaired (staying in community)
                    elif state1['repair'] == 1 and state2['repair'] == 1:
                        persist_prob = 0.9
                    else:
                        persist_prob = base_persist
                    
                    if np.random.random() < persist_prob:
                        current_network.append({
                            'household_id_1': hh1,
                            'household_id_2': hh2,
                            'link_type': 2,
                            'timestep': t
                        })
            
            # NEW link formation (small probability)
            for i, hh1 in enumerate(self.household_ids):
                for hh2 in self.household_ids[i+1:]:
                    if (hh1, hh2) not in prev_network and (hh2, hh1) not in prev_network:
                        
                        # Get states
                        state1 = states_df[(states_df['household_id'] == hh1) & (states_df['timestep'] == t)].iloc[0]
                        state2 = states_df[(states_df['household_id'] == hh2) & (states_df['timestep'] == t)].iloc[0]
                        
                        # Small chance of new bridging links if both staying/repairing
                        if (state1['repair'] == 1 and state2['repair'] == 1 and 
                            state1['vacant'] == 0 and state2['vacant'] == 0):
                            if np.random.random() < 0.008:  # 0.8% chance
                                current_network.append({
                                    'household_id_1': hh1,
                                    'household_id_2': hh2,
                                    'link_type': 2,
                                    'timestep': t
                                })
            
            all_networks.extend(current_network)
        
        return pd.DataFrame(all_networks)
    
    def apply_observation_model(self, complete_network_df):
        """Apply observation model with more realistic patterns."""
        observed_networks = []
        
        for _, link in complete_network_df.iterrows():
            if link['link_type'] == 1:  # Bonding
                base_observation_prob = 0.75  # High observation rate for close relationships
            else:  # Bridging
                base_observation_prob = 0.55  # Moderate observation rate for casual relationships
            
            # Apply global observation rate adjustment
            final_observation_prob = base_observation_prob * (self.observation_rate / 0.3)
            final_observation_prob = min(final_observation_prob, 0.9)  # Cap at 90%
            
            if np.random.random() < final_observation_prob:
                observed_networks.append({
                    'household_id_1': link['household_id_1'],
                    'household_id_2': link['household_id_2'],
                    'link_type': link['link_type'],
                    'timestep': link['timestep']
                })
        
        return pd.DataFrame(observed_networks)
    
    def generate_all_data(self, output_dir='syn_data_v2'):
        """Generate all data files with improved patterns."""
        print("Generating improved synthetic post-disaster recovery data (Version 2)...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data with improved patterns
        print("1. Generating clustered household locations...")
        locations_df = self.generate_household_locations()
        locations_df.to_csv(f'{output_dir}/household_locations.csv', index=False)
        
        print("2. Generating structured household features...")
        features_df = self.generate_household_features()
        features_df.to_csv(f'{output_dir}/household_features.csv', index=False)
        
        print("3. Generating feature-dependent network structure...")
        initial_network = self.generate_initial_network(locations_df, features_df)
        
        print("4. Simulating decision dynamics with strong network effects...")
        states_df = self.simulate_state_evolution(features_df, initial_network)
        states_df.to_csv(f'{output_dir}/household_states.csv', index=False)
        
        print("5. Simulating realistic network evolution...")
        complete_network_df = self.simulate_network_evolution_complete(initial_network, states_df)
        complete_network_df.to_csv(f'{output_dir}/ground_truth_network.csv', index=False)
        
        print("6. Applying realistic observation model...")
        observed_network_df = self.apply_observation_model(complete_network_df)
        observed_network_df.to_csv(f'{output_dir}/observed_network.csv', index=False)
        
        # Enhanced summary statistics
        print("\n=== IMPROVED DATA GENERATION SUMMARY ===")
        print(f"Number of households: {self.n_households}")
        print(f"Number of timesteps: {self.n_timesteps}")
        print(f"Total possible links: {self.n_households * (self.n_households - 1) // 2}")
        
        # Initial network density
        initial_links = len(initial_network)
        total_possible = self.n_households * (self.n_households - 1) // 2
        print(f"Initial network density: {initial_links/total_possible:.1%} ({initial_links} links)")
        
        # Network statistics
        total_ground_truth_edges = len(complete_network_df)
        total_observed_edges = len(observed_network_df)
        observation_rate_actual = total_observed_edges / total_ground_truth_edges if total_ground_truth_edges > 0 else 0
        
        print(f"Total ground truth network entries: {total_ground_truth_edges}")
        print(f"Total observed network entries: {total_observed_edges}")
        print(f"Actual observation rate: {observation_rate_actual:.1%}")
        
        # Link type distribution
        gt_bonding = len(complete_network_df[complete_network_df['link_type'] == 1])
        gt_bridging = len(complete_network_df[complete_network_df['link_type'] == 2])
        obs_bonding = len(observed_network_df[observed_network_df['link_type'] == 1])
        obs_bridging = len(observed_network_df[observed_network_df['link_type'] == 2])
        
        print(f"\nGround truth links - Bonding: {gt_bonding}, Bridging: {gt_bridging}")
        print(f"Observed links - Bonding: {obs_bonding}, Bridging: {obs_bridging}")
        
        if gt_bonding > 0:
            print(f"Bonding observation rate: {obs_bonding/gt_bonding:.1%}")
        if gt_bridging > 0:
            print(f"Bridging observation rate: {obs_bridging/gt_bridging:.1%}")
        
        # Decision statistics
        final_states = states_df[states_df['timestep'] == self.n_timesteps - 1]
        print(f"\nFinal decision statistics:")
        print(f"Vacant (relocated): {final_states['vacant'].sum()} ({final_states['vacant'].mean():.1%})")
        print(f"Repair: {final_states['repair'].sum()} ({final_states['repair'].mean():.1%})")
        print(f"Sell: {final_states['sell'].sum()} ({final_states['sell'].mean():.1%})")
        print(f"No decision: {len(final_states) - final_states[['vacant', 'repair', 'sell']].sum().sum()}")
        
        print(f"\nImproved data files saved to '{output_dir}/' directory")
        
        return {
            'locations': locations_df,
            'features': features_df,
            'states': states_df,
            'ground_truth_network': complete_network_df,
            'observed_network': observed_network_df
        }


# Generate the improved data
if __name__ == "__main__":
    # Create generator with same parameters as original but improved patterns
    generator = DisasterRecoveryDataGeneratorV2(
        n_households=50,       # Same size for comparison
        n_timesteps=25,        # Same timeline
        observation_rate=0.3   # Same observation rate
    )
    
    # Generate improved data
    data = generator.generate_all_data(output_dir='data_v2')
    
    print("\n=== SAMPLE IMPROVED DATA PREVIEW ===")
    print("\nHousehold Features (first 3 rows):")
    print(data['features'].head(3))
    
    print("\nGround Truth Network (first 10 rows):")
    print(data['ground_truth_network'].head(10))
    
    print("\nObserved Network (first 10 rows):")
    print(data['observed_network'].head(10))