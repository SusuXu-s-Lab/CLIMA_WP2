import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class DisasterRecoveryDataGenerator:
    def __init__(self, n_households=200, n_timesteps=25, observation_rate=0.3):
        """
        Generate synthetic data for post-disaster recovery network analysis
        
        Parameters:
        - n_households: Number of households in the community
        - n_timesteps: Number of time steps to simulate (0 to T)
        - observation_rate: Fraction of actual network links that are observed
        """
        self.n_households = n_households
        self.n_timesteps = n_timesteps
        self.observation_rate = observation_rate
        
        # Generate household IDs
        self.household_ids = list(range(1, n_households + 1))
        
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
                'education_level': np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2]),  # 1=HS, 2=Some college, 3=Bachelor, 4=Graduate
                'years_in_community': np.random.exponential(8),
                'damage_severity': np.random.uniform(0.1, 1.0)  # Proportion of home damaged
            })
        
        return pd.DataFrame(features)
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate approximate distance between two points in km"""
        # Simple Euclidean distance approximation
        lat_diff = (lat2 - lat1) * 111  # ~111 km per degree latitude
        lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def generate_initial_network(self, locations_df, features_df):
        """Generate initial network structure based on proximity and similarity"""
        network_links = []
        
        # Create distance matrix
        for i, hh1 in enumerate(self.household_ids):
            for j, hh2 in enumerate(self.household_ids[i+1:], i+1):
                
                # Get locations
                loc1 = locations_df[locations_df['household_id'] == hh1].iloc[0]
                loc2 = locations_df[locations_df['household_id'] == hh2].iloc[0]
                
                distance = self.calculate_distance(
                    loc1['latitude'], loc1['longitude'],
                    loc2['latitude'], loc2['longitude']
                )
                
                # Get features for similarity calculation
                feat1 = features_df[features_df['household_id'] == hh1].iloc[0]
                feat2 = features_df[features_df['household_id'] == hh2].iloc[0]
                
                # Calculate demographic similarity
                income_sim = 1 - abs(feat1['income'] - feat2['income']) / max(feat1['income'], feat2['income'])
                age_sim = 1 - abs(feat1['age_head'] - feat2['age_head']) / 50
                edu_sim = 1 - abs(feat1['education_level'] - feat2['education_level']) / 3
                
                similarity = (income_sim + age_sim + edu_sim) / 3
                
                # Determine link probability based on distance and similarity
                # Bonding links: high similarity, any distance (family/close friends)
                bonding_prob = 0.02 * similarity**2
                
                # Bridging links: proximity-based (neighbors)
                if distance < 0.5:  # Within 500m
                    bridging_prob = 0.3 * (1 - distance/0.5)
                elif distance < 2.0:  # Within 2km
                    bridging_prob = 0.1 * (1 - distance/2.0)
                else:
                    bridging_prob = 0.01
                
                # Decide link type
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
        """Simulate household decision evolution over time"""
        # Initialize states - all households start undecided
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
        
        # Simulate evolution
        for t in range(1, self.n_timesteps):
            # Create network at current time (simplified - keep initial for now)
            current_network = {(link['household_id_1'], link['household_id_2']): link['link_type'] 
                             for link in initial_network}
            
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
                
                # Calculate self-activation probabilities
                damage = hh_features['damage_severity']
                income = hh_features['income']
                insurance = hh_features['insurance_coverage']
                
                # Higher damage -> more likely to relocate or sell
                # Higher income + insurance -> more likely to repair
                base_vacant_prob = 0.02 + 0.05 * damage * (1 - insurance)
                base_repair_prob = 0.03 + 0.06 * insurance * (income / 100000)
                base_sell_prob = 0.015 + 0.03 * damage
                
                # Add social influence
                connected_neighbors = []
                for (hh1, hh2), link_type in current_network.items():
                    if hh1 == hh_id:
                        connected_neighbors.append((hh2, link_type))
                    elif hh2 == hh_id:
                        connected_neighbors.append((hh1, link_type))
                
                influence_vacant = 0
                influence_repair = 0
                influence_sell = 0
                
                for neighbor_id, link_type in connected_neighbors:
                    neighbor_state = current_states[neighbor_id]
                    
                    # Influence strength depends on link type
                    influence_strength = 0.3 if link_type == 1 else 0.1  # Bonding vs Bridging
                    
                    if neighbor_state['vacant'] == 1:
                        influence_vacant += influence_strength * 0.4
                    if neighbor_state['repair'] == 1:
                        influence_repair += influence_strength * 0.5
                    if neighbor_state['sell'] == 1:
                        influence_sell += influence_strength * 0.3
                
                # Calculate final probabilities
                vacant_prob = min(1.0, base_vacant_prob + influence_vacant)
                repair_prob = min(1.0, base_repair_prob + influence_repair)
                sell_prob = min(1.0, base_sell_prob + influence_sell)
                
                # Normalize probabilities (ensure they don't exceed 1 total)
                total_prob = vacant_prob + repair_prob + sell_prob
                if total_prob > 0.8:  # Leave some probability for no decision
                    vacant_prob *= 0.8 / total_prob
                    repair_prob *= 0.8 / total_prob
                    sell_prob *= 0.8 / total_prob
                
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
        """Simulate complete network evolution over time (ground truth)"""
        all_networks = []
        
        # Add initial network
        all_networks.extend(initial_network)
        
        # Simulate evolution
        for t in range(1, self.n_timesteps):
            current_network = []
            
            # Get previous network
            prev_network = {(link['household_id_1'], link['household_id_2']): link['link_type'] 
                           for link in all_networks if link['timestep'] == t-1}
            
            for (hh1, hh2), prev_link_type in prev_network.items():
                # Get current states
                state1 = states_df[(states_df['household_id'] == hh1) & (states_df['timestep'] == t)].iloc[0]
                state2 = states_df[(states_df['household_id'] == hh2) & (states_df['timestep'] == t)].iloc[0]
                
                # Bonding links persist
                if prev_link_type == 1:
                    current_network.append({
                        'household_id_1': hh1,
                        'household_id_2': hh2,
                        'link_type': 1,
                        'timestep': t
                    })
                
                # Bridging links may decay if someone moved
                elif prev_link_type == 2:
                    decay_prob = 0.05  # Base decay
                    if state1['vacant'] == 1 or state2['vacant'] == 1:
                        decay_prob = 0.7  # High decay if someone moved
                    
                    if np.random.random() > decay_prob:
                        current_network.append({
                            'household_id_1': hh1,
                            'household_id_2': hh2,
                            'link_type': 2,
                            'timestep': t
                        })
            
            # Small probability of new bridging links forming
            for i, hh1 in enumerate(self.household_ids):
                for hh2 in self.household_ids[i+1:]:
                    if (hh1, hh2) not in prev_network and (hh2, hh1) not in prev_network:
                        # Very small probability of new link
                        if np.random.random() < 0.005:
                            current_network.append({
                                'household_id_1': hh1,
                                'household_id_2': hh2,
                                'link_type': 2,
                                'timestep': t
                            })
            
            all_networks.extend(current_network)
        
        return pd.DataFrame(all_networks)
    
    def apply_observation_model(self, complete_network_df):
        """Apply observation model to complete network to get partial observations"""
        observed_networks = []
        
        for _, link in complete_network_df.iterrows():
            if link['link_type'] == 1:  # Bonding
                base_observation_prob = 0.7  # Higher observation rate for strong ties
            else:  # Bridging
                base_observation_prob = 0.4  # Lower observation rate for weak ties
            
            # Apply global observation rate
            final_observation_prob = base_observation_prob * (self.observation_rate / 0.3)
            
            if np.random.random() < final_observation_prob:
                observed_networks.append({
                    'household_id_1': link['household_id_1'],
                    'household_id_2': link['household_id_2'],
                    'link_type': link['link_type'],
                    'timestep': link['timestep']
                })
        
        return pd.DataFrame(observed_networks)
    
    def generate_all_data(self, output_dir='data'):
        """Generate all data files"""
        print("Generating synthetic post-disaster recovery data...")
        
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
        
        # Generate initial network
        print("3. Generating initial network structure...")
        initial_network = self.generate_initial_network(locations_df, features_df)
        
        # Simulate state evolution
        print("4. Simulating household decision dynamics...")
        states_df = self.simulate_state_evolution(features_df, initial_network)
        states_df.to_csv(f'{output_dir}/household_states.csv', index=False)
        
        # Simulate complete network evolution (ground truth)
        print("5. Simulating complete network evolution (ground truth)...")
        complete_network_df = self.simulate_network_evolution_complete(initial_network, states_df)
        complete_network_df.to_csv(f'{output_dir}/ground_truth_network.csv', index=False)
        
        # Apply observation model to get partial observations
        print("6. Applying observation model to get partial network observations...")
        observed_network_df = self.apply_observation_model(complete_network_df)
        observed_network_df.to_csv(f'{output_dir}/observed_network.csv', index=False)
        
        # Generate summary statistics
        print("\n=== Data Generation Summary ===")
        print(f"Number of households: {self.n_households}")
        print(f"Number of timesteps: {self.n_timesteps}")
        print(f"Total possible links: {self.n_households * (self.n_households - 1) // 2}")
        print(f"Initial observed links: {len([l for l in initial_network if l])}")
        
        # Network statistics
        total_ground_truth_edges = len(complete_network_df)
        total_observed_edges = len(observed_network_df)
        observation_rate_actual = total_observed_edges / total_ground_truth_edges if total_ground_truth_edges > 0 else 0
        
        print(f"Total ground truth network entries: {total_ground_truth_edges}")
        print(f"Total observed network entries: {total_observed_edges}")
        print(f"Actual observation rate: {observation_rate_actual:.1%}")
        
        # Ground truth network statistics by type
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
    # Set working directory
    import os
    os.chdir('/Users/susangao/Desktop/CLIMA/CODE/data') 
    
    # Create generator with realistic parameters
    generator = DisasterRecoveryDataGenerator(
        n_households=200,      # Medium-sized community
        n_timesteps=25,        # ~2 years with monthly observations
        observation_rate=0.3   # 30% of actual links observed
    )
    
    # Generate all data
    data = generator.generate_all_data(output_dir='data_format_sample')
    
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