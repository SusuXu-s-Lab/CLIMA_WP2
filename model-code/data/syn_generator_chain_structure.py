import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path

class ChainNetworkTestGenerator:
    """
    Generate super simple test case: 5 households in a chain (A-B-C-D-E)
    Perfect sequential activation to test variational inference
    """
    
    def __init__(self):
        self.n_households = 5
        self.n_timesteps = 6  # t=0,1,2,3,4,5
        self.household_ids = [1, 2, 3, 4, 5]  # A, B, C, D, E
        
    def generate_all_data(self, output_dir='chain_test_data'):
        """Generate minimal test case"""
        print("=== Generating Chain Network Test Case ===")
        print("Structure: A-B-C-D-E (5 households, chain topology)")
        print("Activation: Sequential, one per timestep")
        print("Decision: Only 'vacant' decision, others stay 0")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Simple locations (evenly spaced on a line)
        locations_df = self._generate_locations()
        locations_df.to_csv(f'{output_dir}/household_locations.csv', index=False)
        
        # 2. Identical features (to remove confounding factors)
        features_df = self._generate_features()
        features_df.to_csv(f'{output_dir}/household_features.csv', index=False)
        
        # 3. Perfect sequential states (deterministic)
        states_df = self._generate_states()
        states_df.to_csv(f'{output_dir}/household_states.csv', index=False)
        
        # 4. Ground truth network (chain structure, static)
        gt_network_df = self._generate_ground_truth_network()
        gt_network_df.to_csv(f'{output_dir}/ground_truth_network.csv', index=False)
        
        # 5. Observed network (partial observation)
        obs_network_df = self._generate_observed_network(gt_network_df)
        obs_network_df.to_csv(f'{output_dir}/observed_network.csv', index=False)
        
        self._print_summary(states_df, gt_network_df, obs_network_df)
        
        return {
            'locations': locations_df,
            'features': features_df,
            'states': states_df,
            'ground_truth_network': gt_network_df,
            'observed_network': obs_network_df
        }
    
    def _generate_locations(self):
        """Evenly spaced locations on a line"""
        locations = []
        base_lat = 40.7128
        base_lon = -74.0060
        
        for i, hh_id in enumerate(self.household_ids):
            # Evenly spaced along longitude
            lat = base_lat
            lon = base_lon + i * 0.01  # 1km apart
            
            locations.append({
                'household_id': hh_id,
                'latitude': lat,
                'longitude': lon
            })
        
        return pd.DataFrame(locations)
    
    def _generate_features(self):
        """Identical features to remove confounding"""
        features = []
        
        for hh_id in self.household_ids:
            features.append({
                'household_id': hh_id,
                'income': 50000.0,  # Same for all
                'property_value': 200000.0,  # Same for all
                'insurance_coverage': 1,  # Same for all
                'household_size': 3,  # Same for all
                'age_head': 40.0,  # Same for all
                'education_level': 2,  # Same for all
                'years_in_community': 5.0,  # Same for all
                'damage_severity': 0.5  # Same for all
            })
        
        return pd.DataFrame(features)
    
    def _generate_states(self):
        """Perfect sequential activation pattern"""
        states = []
        
        # Activation sequence: A(t=1), B(t=2), C(t=3), D(t=4), E(t=5)
        activation_times = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        
        for t in range(self.n_timesteps):
            for hh_id in self.household_ids:
                # Only 'vacant' decision activates, others stay 0
                vacant = 1 if t >= activation_times[hh_id] else 0
                repair = 0  # Always 0
                sell = 0    # Always 0
                
                states.append({
                    'household_id': hh_id,
                    'timestep': t,
                    'vacant': vacant,
                    'repair': repair,
                    'sell': sell
                })
        
        return pd.DataFrame(states)
    
    def _generate_ground_truth_network(self):
        """Static chain structure A-B-C-D-E"""
        network_links = []
        
        # Chain connections: (1,2), (2,3), (3,4), (4,5)
        chain_pairs = [(1, 2), (2, 3), (3, 4), (4, 5)]
        
        for t in range(self.n_timesteps):
            for hh1, hh2 in chain_pairs:
                network_links.append({
                    'household_id_1': hh1,
                    'household_id_2': hh2,
                    'link_type': 2,  # All bridging links
                    'timestep': t
                })
        
        return pd.DataFrame(network_links)
    
    def _generate_observed_network(self, gt_network_df):
        """Partial observation: hide some links to test inference"""
        observed_links = []
        
        # Hide specific links to create interesting inference problem
        hidden_pairs = {(2, 3), (4, 5)}  # Hide B-C and D-E connections
        
        for _, row in gt_network_df.iterrows():
            pair = (row['household_id_1'], row['household_id_2'])
            
            # Keep observed if not in hidden set
            if pair not in hidden_pairs:
                observed_links.append({
                    'household_id_1': row['household_id_1'],
                    'household_id_2': row['household_id_2'],
                    'link_type': row['link_type'],
                    'timestep': row['timestep']
                })
        
        return pd.DataFrame(observed_links)
    
    def _print_summary(self, states_df, gt_network_df, obs_network_df):
        """Print test case summary"""
        print(f"\n=== Chain Test Case Summary ===")
        print(f"Households: {self.n_households} (IDs: {self.household_ids})")
        print(f"Timesteps: {self.n_timesteps} (0 to {self.n_timesteps-1})")
        
        print(f"\nNetwork Structure:")
        print(f"Ground truth links: {len(gt_network_df)} total")
        print(f"Observed links: {len(obs_network_df)} total")
        print(f"Hidden links: {len(gt_network_df) - len(obs_network_df)} total")
        
        # Show which links are hidden
        gt_pairs = set()
        obs_pairs = set()
        
        for _, row in gt_network_df.iterrows():
            if row['timestep'] == 0:  # Just look at t=0
                gt_pairs.add((row['household_id_1'], row['household_id_2']))
        
        for _, row in obs_network_df.iterrows():
            if row['timestep'] == 0:  # Just look at t=0
                obs_pairs.add((row['household_id_1'], row['household_id_2']))
        
        hidden_pairs = gt_pairs - obs_pairs
        print(f"Observed pairs: {sorted(obs_pairs)}")
        print(f"Hidden pairs: {sorted(hidden_pairs)}")
        
        print(f"\nActivation Pattern (vacant decision only):")
        final_states = states_df[states_df['timestep'] == self.n_timesteps - 1]
        for _, row in final_states.iterrows():
            hh_id = row['household_id']
            vacant = row['vacant']
            print(f"Household {hh_id}: vacant={vacant}")
        
        print(f"\nExpected Model Behavior:")
        print(f"1. Should infer hidden links (2,3) and (4,5)")
        print(f"2. Should learn strong social influence along chain")
        print(f"3. Should predict sequential activation pattern")
        print(f"4. Variational posterior should be confident about chain structure")


# Additional monitoring tools
class ChainTestMonitor:
    """Monitor training progress for chain test case"""
    
    def __init__(self, data_dir='chain_test_data'):
        self.data_dir = Path(data_dir)
        
    def analyze_initial_state(self, model_components):
        """Analyze model state before training"""
        print("=== Initial Model State Analysis ===")
        
        # Check parameter initialization
        posterior, network_evolution, state_transition = model_components
        
        print(f"Network evolution parameters:")
        print(f"  alpha_0: {network_evolution.alpha_0.item():.4f}")
        
        print(f"Observation parameters:")
        print(f"  rho_1: {model_components[-1].rho_1.item():.4f}")
        print(f"  rho_2: {model_components[-1].rho_2.item():.4f}")
    
    def analyze_probabilities(self, marginal_probs, conditional_probs, timestep=0):
        """Analyze variational probabilities"""
        print(f"\n=== Probability Analysis (t={timestep}) ===")
        
        for pair_key, prob in marginal_probs.items():
            if f"_{timestep}" in pair_key:
                i, j, t = pair_key.split('_')
                prob_np = prob.detach().numpy()
                print(f"Pair ({i},{j}): π̄=[{prob_np[0]:.3f}, {prob_np[1]:.3f}, {prob_np[2]:.3f}]")
                
                # Also show conditional if available
                if pair_key in conditional_probs:
                    cond_prob = conditional_probs[pair_key].detach().numpy()
                    print(f"  Conditional π(t|k'):")
                    for k_prev in range(3):
                        print(f"    k'={k_prev}: [{cond_prob[k_prev,0]:.3f}, {cond_prob[k_prev,1]:.3f}, {cond_prob[k_prev,2]:.3f}]")
    
    def analyze_influence_patterns(self, state_transition, features, states, distances, network_data, gumbel_samples, timestep=1):
        """Analyze learned influence patterns"""
        print(f"\n=== Influence Analysis (t={timestep}) ===")
        
        # Test influence from household 1 to household 2 (known connection)
        # Assume household 1 is active, test influence on household 2
        
        for decision_type in range(3):
            print(f"\nDecision type {decision_type}:")
            
            # Test all pairs
            for i in range(5):
                for j in range(5):
                    if i != j:
                        try:
                            # Create test scenario
                            test_states = states.clone()
                            test_states[i, timestep, decision_type] = 1  # i is active
                            test_states[j, timestep, decision_type] = 0  # j is inactive
                            
                            prob = state_transition.compute_activation_probability(
                                household_idx=torch.tensor([j]),
                                decision_type=decision_type,
                                features=features,
                                states=test_states,
                                distances=distances,
                                network_data=network_data,
                                gumbel_samples=gumbel_samples[0] if gumbel_samples else {},
                                time=timestep
                            )
                            
                            print(f"  {i}→{j}: {prob.item():.4f}")
                        except:
                            print(f"  {i}→{j}: ERROR")
    
    def monitor_training_step(self, epoch, metrics, marginal_probs=None):
        """Monitor each training step"""
        if epoch % 10 == 0:  # Every 10 epochs
            print(f"\n=== Epoch {epoch} ===")
            print(f"ELBO: {metrics['total_elbo']:.4f}")
            print(f"  State: {metrics['state_likelihood']:.4f}")
            print(f"  Observation: {metrics['observation_likelihood']:.4f}")
            print(f"  Prior: {metrics['prior_likelihood']:.4f}")
            print(f"  Entropy: {metrics['posterior_entropy']:.4f}")
            print(f"Temperature: {metrics['temperature']:.4f}")
            
            # Show parameter evolution
            print(f"Parameters:")
            print(f"  rho_1: {metrics['rho_1']:.4f}")
            print(f"  rho_2: {metrics['rho_2']:.4f}")
            
            # Show key probabilities if available
            if marginal_probs:
                self.analyze_probabilities(marginal_probs, {}, timestep=0)


if __name__ == "__main__":
    # Generate test case
    generator = ChainNetworkTestGenerator()
    data = generator.generate_all_data(output_dir='CODE 5/data/syn_data_chain_node')
    
    print("\n=== Generated Files ===")
    for filename in ['household_locations.csv', 'household_features.csv', 
                     'household_states.csv', 'ground_truth_network.csv', 
                     'observed_network.csv']:
        print(f"- {filename}")
    
    print("\n=== Next Steps ===")
    print("1. Load this data with your DataLoader")
    print("2. Initialize model components")
    print("3. Use ChainTestMonitor to track training")
    print("4. Expected outcome: Model should infer missing chain links")