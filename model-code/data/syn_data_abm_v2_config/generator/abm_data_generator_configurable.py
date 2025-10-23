import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import os
from tqdm import tqdm
import warnings
from abm_config import ABMScenarioConfig

warnings.filterwarnings("ignore")

@dataclass
class AgentConfig:
    """Agent role configuration"""
    proportion: float
    self_motivation: str  # 'high', 'medium', 'low'
    social_responsiveness: str  # 'high', 'medium', 'low', 'none'
    network_activity: str  # 'high', 'medium', 'low'
    purpose: str

class HouseholdAgent:
    """Individual household agent with role-based behavior"""
    
    def __init__(self, household_id: int, agent_type: str, features: Dict):
        self.household_id = household_id
        self.agent_type = agent_type
        self.features = features
        
        # Decision history: {decision_type: [values_over_time]}
        self.decision_history = {
            'repair': [0],  # t=0 initial state
            'vacant': [0],
            'sell': [0]
        }
        
        # Network connections: {neighbor_id: link_type}
        self.connections = {}
        
        # Role-specific parameters
        self.role_params = self._get_role_parameters()
        
    def _get_role_parameters(self) -> Dict:
        """Get role-specific parameter multipliers"""
        # role_configs = {
        #     'early_adopter': {
        #         'self_motivation_range': (3.0, 5.0),
        #         'social_responsiveness_range': (0.1, 0.3),
        #         'network_formation_tendency': 0.6
        #     },
        #     'social_follower': {
        #         'self_motivation_range': (0.1, 0.3),
        #         'social_responsiveness_range': (2.0, 4.0),
        #         'network_formation_tendency': 0.9
        #     },
        #     'resistant': {
        #         'self_motivation_range': (0.01, 0.05),
        #         'social_responsiveness_range': (0.02, 0.08),
        #         'network_formation_tendency': 0.2
        #     },
        #     'isolated': {
        #         'self_motivation_range': (0.1, 0.5),
        #         'social_responsiveness_range': (0.0, 0.02),
        #         'network_formation_tendency': 0.05
        #     }
        # }

        # role_configs = {
        #     'early_adopter': {
        #         'self_motivation_range': (1.2, 1.8),      # 改为 1.2-1.8x
        #         'social_responsiveness_range': (0.8, 1.2), # 改为 0.8-1.2x
        #         'network_formation_tendency': 0.6
        #     },
        #     'social_follower': {
        #         'self_motivation_range': (0.3, 0.6),      # 改为 0.3-0.6x
        #         'social_responsiveness_range': (1.5, 2.5), # 改为 1.5-2.5x
        #         'network_formation_tendency': 0.9
        #     },
        #     'resistant': {
        #         'self_motivation_range': (0.1, 0.3),      # 改为 0.1-0.3x
        #         'social_responsiveness_range': (0.1, 0.3), # 改为 0.1-0.3x
        #         'network_formation_tendency': 0.2
        #     },
        #     'isolated': {
        #         'self_motivation_range': (0.5, 1.0),      # 改为 0.5-1.0x
        #         'social_responsiveness_range': (0.05, 0.15), # 改为 0.05-0.15x
        #         'network_formation_tendency': 0.05
        #     }
        # }
        
        # 在 HouseholdAgent._get_role_parameters() 中
        role_configs = {
            'early_adopter': {
                'self_motivation_range': (1.5, 2.5),      # 1.5-2.5x
                'social_responsiveness_range': (0.8, 1.2), # 0.8-1.2x
                'network_formation_tendency': 0.6
            },
            'social_follower': {
                'self_motivation_range': (0.3, 0.6),      # 0.3-0.6x
                'social_responsiveness_range': (2.0, 3.0), # 2.0-3.0x
                'network_formation_tendency': 0.9
            },
            'resistant': {
                'self_motivation_range': (0.05, 0.15),    # 改：0.05-0.15x (原来0.01-0.05)
                'social_responsiveness_range': (0.2, 0.5), # 改：0.2-0.5x (原来0.02-0.08)
                'network_formation_tendency': 0.2
            },
            'isolated': {
                'self_motivation_range': (0.5, 1.0),      # 0.5-1.0x
                'social_responsiveness_range': (0.05, 0.15), # 0.05-0.15x
                'network_formation_tendency': 0.05
            }
        }

        config = role_configs[self.agent_type]
        return {
            'self_multiplier': np.random.uniform(*config['self_motivation_range']),
            'social_multiplier': np.random.uniform(*config['social_responsiveness_range']),
            'network_tendency': config['network_formation_tendency']
        }

class ABMDataGenerator:
    """Agent-based model for generating attribution test data with configurable scenarios"""
    
    def __init__(self, config: ABMScenarioConfig):
        """
        Initialize ABM data generator with configuration
        
        Args:
            config: ABMScenarioConfig object with all scenario parameters
        """
        self.config = config
        np.random.seed(config.random_seed)
        
        self.n_households = config.n_households
        self.n_timesteps = config.n_timesteps
        self.household_ids = list(range(1, self.n_households + 1))
        
        # Agent type distribution from config
        self.agent_types = {
            'early_adopter': AgentConfig(
                config.early_adopter_ratio, 'high', 'low', 'medium', 
                'pure self-activation evidence'),
            'social_follower': AgentConfig(
                config.social_follower_ratio, 'low', 'high', 'high', 
                'social influence evidence'),
            'resistant': AgentConfig(
                config.resistant_ratio, 'low', 'low', 'low', 
                'negative evidence'),
            'isolated': AgentConfig(
                config.isolated_ratio, 'medium', 'none', 'minimal', 
                'self-activation controls')
        }
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Data storage
        self.network_history = []
        self.probability_log = []
        self.p_self_all_values = []
        self.p_ji_all_values = []
        
        # Network density calibration
        self._compute_network_scale_factor()

        # ========== NEW: Time dynamics functions ==========
    
    def _compute_time_effect(self, decision_type: str, t: int) -> float:
        """
        Compute time-dependent effect for decision activation (bell curve shape)
        
        Returns a multiplier in [0, 1] representing how "active" this decision type
        should be at time t. Before start_time it's near 0, peaks at peak_time, 
        then decays.
        
        Args:
            decision_type: 'repair', 'vacant', or 'sell'
            t: current timestep
            
        Returns:
            float: time effect multiplier in [0, 1]
        """
        # Get parameters from config
        if decision_type == 'repair':
            start_time = self.config.repair_start_time
            peak_time = self.config.repair_peak_time
        elif decision_type == 'vacant':
            start_time = self.config.vacant_start_time
            peak_time = self.config.vacant_peak_time
        elif decision_type == 'sell':
            start_time = self.config.sell_start_time
            peak_time = self.config.sell_peak_time
        else:
            return 1.0  # Unknown type, no effect
        
        # Before start time: very low activation (but not zero, allowing rare early cases)
        if t < start_time:
            return 0.30  # 5% of normal probability
        
        # Growth phase: from start_time to peak_time
        if t <= peak_time:
            # Sigmoid-like growth
            progress = (t - start_time) / max(1, peak_time - start_time)
            return 0.05 + 0.95 * (1 / (1 + np.exp(-5 * (progress - 0.5))))
        
        # Decay phase: after peak_time
        else:
            # Exponential decay
            time_after_peak = t - peak_time
            decay_rate = 0.1  # Adjust this to control how fast it decays
            return np.exp(-decay_rate * time_after_peak)
    
    def _get_decision_order(self) -> list:
        """
        Get the order in which decisions should be evaluated at each timestep.
        
        Returns decisions in order of typical causal flow: repair -> sell -> vacant
        """
        return ['repair', 'vacant', 'sell']
        
    def _compute_network_scale_factor(self):
        """
        Compute global scale factor to achieve target average degree
        Uses simple scale factor approach (A2 strategy)
        """
        # Calculate expected total degree without scaling
        expected_degree = 0
        for agent_type, config in self.agent_types.items():
            n_agents = int(self.n_households * config.proportion)
            
            # Base lambda values for each role (from original code)
            if agent_type == 'isolated':
                base_lambda = 0.3
            elif agent_type == 'social_follower':
                base_lambda = 6
            elif agent_type == 'early_adopter':
                base_lambda = 4
            else:  # resistant
                base_lambda = 2
            
            expected_degree += n_agents * base_lambda
        
        expected_avg_degree = expected_degree / self.n_households
        
        # Compute scale factor
        if expected_avg_degree > 0:
            self.network_scale_factor = self.config.target_avg_degree / expected_avg_degree
        else:
            self.network_scale_factor = 1.0
        
        print(f"\nNetwork Calibration:")
        print(f"  Expected avg degree (unscaled): {expected_avg_degree:.2f}")
        print(f"  Target avg degree: {self.config.target_avg_degree}")
        print(f"  Scale factor: {self.network_scale_factor:.3f}")
    
    def _initialize_agents(self) -> Dict[int, HouseholdAgent]:
        """Initialize all household agents with features and roles"""
        agents = {}
        
        # Assign agent types based on proportions from config
        type_assignments = self._assign_agent_types()
        
        for hid in self.household_ids:
            # Generate features
            features = self._generate_household_features(hid)
            
            # Create agent
            agent_type = type_assignments[hid]
            agents[hid] = HouseholdAgent(hid, agent_type, features)
            
        return agents
    
    def _assign_agent_types(self) -> Dict[int, str]:
        """Assign agent types based on proportions from config"""
        assignments = {}
        remaining_ids = self.household_ids.copy()
        
        for agent_type, config in self.agent_types.items():
            n_agents = int(self.n_households * config.proportion)
            if agent_type == 'resistant':  # Assign remaining to resistant
                selected_ids = remaining_ids
            else:
                if len(remaining_ids) >= n_agents:
                    selected_ids = np.random.choice(remaining_ids, n_agents, replace=False)
                    remaining_ids = [hid for hid in remaining_ids if hid not in selected_ids]
                else:
                    selected_ids = remaining_ids
                    remaining_ids = []
            
            for hid in selected_ids:
                assignments[hid] = agent_type
                
        return assignments
    
    def _generate_household_features(self, household_id: int) -> Dict:
        """Generate feature vector for a household"""
        # Create communities (first 6 chars of imaginary geohash)
        community_id = np.random.randint(0, 20)  # 20 communities
        
        # Community-level effects
        high_damage_communities = np.random.choice(20, 3, replace=False)
        high_value_communities = np.random.choice(20, 3, replace=False)
        
        # Generate features
        features = {
            'community_id': community_id,
            'building_value': np.clip(np.random.normal(
                0.7 if community_id in high_value_communities else 0.5, 0.15), 0, 1),
            'income': np.clip(np.random.beta(
                2 if community_id < 10 else 0.5, 2), 0, 1),
            'damage_level': np.random.choice(
                [0.0, 0.25, 0.5, 0.75, 1.0],
                p=[0.1, 0.1, 0.2, 0.3, 0.3] if community_id in high_damage_communities 
                  else [0.7, 0.1, 0.1, 0.05, 0.05]),
            'population_scaled': np.random.randint(1, 8) / 7.0,
            'age': np.clip(np.random.normal(0.5, 0.2), 0, 1),
            'race': np.random.choice([0.0, 0.25, 0.5, 1.0], p=[0.6, 0.2, 0.1, 0.1]),
            'latitude': 39.0 + np.random.normal(0, 0.1),
            'longitude': -76.6 + np.random.normal(0, 0.1)
        }
        
        return features
    
    def generate_initial_decisions_controlled(self):
        """
        Generate initial decisions at t=0 with controlled seed ratios for each decision type.
        Uses config parameters: target_seed_ratio, repair_seed_ratio, vacant_seed_ratio, sell_seed_ratio
        """
        N = self.n_households
        
        # Calculate total number of seeds and breakdown by decision type
        total_seeds = int(np.round(self.config.target_seed_ratio * N))
        repair_count = int(round(total_seeds * self.config.repair_seed_ratio))
        vacant_count = int(round(total_seeds * self.config.vacant_seed_ratio))
        sell_count = total_seeds - repair_count - vacant_count  # Ensure sum equals total_seeds
        
        print(f"\nInitial seed allocation at t=0:")
        print(f"  Total seeds: {total_seeds} ({self.config.target_seed_ratio:.1%} of {N})")
        print(f"  Repair: {repair_count} ({repair_count/N:.1%})")
        print(f"  Vacant: {vacant_count} ({vacant_count/N:.1%})")
        print(f"  Sell: {sell_count} ({sell_count/N:.1%})")
        
        # Compute tendency scores for each decision type
        seed_scores = []
        for agent in self.agents.values():
            features = agent.features
            
            # Repair tendency
            repair_tendency = (
                2.0 * features['damage_level'] +
                1.0 * features['building_value'] +
                0.5 * features['income']
            ) * agent.role_params['self_multiplier']
            
            # Vacant tendency
            community_damage = np.mean([self.agents[hid].features['damage_level'] 
                                    for hid in self.household_ids 
                                    if self.agents[hid].features['community_id'] == features['community_id']])
            
            vacant_tendency = (
                2.0 * community_damage +
                -1.0 * features['building_value'] +
                -0.8 * features['income']
            ) * agent.role_params['self_multiplier']
            
            # Sell tendency
            sell_tendency = (
                1.0 * features['damage_level'] +
                -0.5 * features['building_value'] +
                -0.3 * features['income']
            ) * agent.role_params['self_multiplier']
            
            seed_scores.append((
                agent.household_id, 
                repair_tendency, 
                vacant_tendency, 
                sell_tendency
            ))
        
        # Build type-specific rankings
        repair_candidates = [(hid, r_tend) for hid, r_tend, _, _ in seed_scores]
        vacant_candidates = [(hid, v_tend) for hid, _, v_tend, _ in seed_scores]
        sell_candidates = [(hid, s_tend) for hid, _, _, s_tend in seed_scores]
        
        repair_candidates.sort(key=lambda x: x[1], reverse=True)
        vacant_candidates.sort(key=lambda x: x[1], reverse=True)
        sell_candidates.sort(key=lambda x: x[1], reverse=True)
        
        assigned = set()
        
        def assign_from_list(candidates, count, decision_key):
            """Assign top candidates to a decision, skipping already assigned households"""
            taken = 0
            for hid, _ in candidates:
                if taken >= count:
                    break
                if hid in assigned:
                    continue
                self.agents[hid].decision_history[decision_key][0] = 1
                assigned.add(hid)
                taken += 1
            return taken
        
        # Allocate in order: sell -> vacant -> repair (to prevent sell from being starved)
        sell_assigned = assign_from_list(sell_candidates, sell_count, 'sell')
        vacant_assigned = assign_from_list(vacant_candidates, vacant_count, 'vacant')
        repair_assigned = assign_from_list(repair_candidates, repair_count, 'repair')
        
        # Backfill if overlaps prevented exact quotas
        remaining = total_seeds - len(assigned)
        if remaining > 0:
            print(f"  Backfilling {remaining} seeds due to overlaps...")
            # Sort by overall tendency
            overall_scores = [(hid, r+v+s) for hid, r, v, s in seed_scores]
            overall_scores.sort(key=lambda x: x[1], reverse=True)
            
            for hid, _ in overall_scores:
                if remaining == 0:
                    break
                if hid in assigned:
                    continue
                
                # Assign to the decision type with highest individual tendency
                agent = self.agents[hid]
                tendencies = {
                    'repair': next(r for h, r, _, _ in seed_scores if h == hid),
                    'vacant': next(v for h, _, v, _ in seed_scores if h == hid),
                    'sell': next(s for h, _, _, s in seed_scores if h == hid)
                }
                best_decision = max(tendencies, key=tendencies.get)
                agent.decision_history[best_decision][0] = 1
                assigned.add(hid)
                remaining -= 1
        
        print(f"  Final allocation: repair={repair_assigned}, vacant={vacant_assigned}, sell={sell_assigned}")
        print(f"  Total assigned: {len(assigned)}\n")
    
    def _initialize_network(self):
        """
        Initialize t=0 network with scaled connection counts
        """
        print(f"\nInitializing network with scale factor: {self.network_scale_factor:.3f}")
        
        for agent in self.agents.values():
            # Base number of connections depends on agent type (scaled)
            if agent.agent_type == 'isolated':
                base_lambda = 0.3
            elif agent.agent_type == 'social_follower':
                base_lambda = 6
            elif agent.agent_type == 'early_adopter':
                base_lambda = 4
            else:  # resistant
                base_lambda = 2
            
            # Apply scale factor
            scaled_lambda = base_lambda * self.network_scale_factor
            n_connections = np.random.poisson(scaled_lambda)
            n_connections = min(n_connections, 10)  # Cap at 10
            
            # Select random other agents based on proximity and similarity
            candidates = []
            for other_id, other_agent in self.agents.items():
                if other_id != agent.household_id and other_id not in agent.connections:
                    base_prob = self._compute_connection_probability(agent, other_agent)
                    
                    # Bidirectional consent: consider other agent's willingness to connect
                    acceptance_prob = other_agent.role_params['network_tendency']
                    final_prob = base_prob * acceptance_prob
                    
                    candidates.append((other_id, final_prob))
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected = candidates[:n_connections]
                
                for neighbor_id, _ in selected:
                    if neighbor_id not in agent.connections:
                        link_type = self._determine_link_type_controlled(agent, self.agents[neighbor_id])
                        agent.connections[neighbor_id] = link_type
                        self.agents[neighbor_id].connections[agent.household_id] = link_type
        
        # # Save initial network
        # self._save_network_state(0)
        
        # Report actual network stats
        total_links = len(self.network_history)
        bonding_links = sum(1 for link in self.network_history if link['link_type'] == 1)
        bridging_links = sum(1 for link in self.network_history if link['link_type'] == 2)
        actual_avg_degree = (2 * total_links) / self.n_households
        actual_bonding_ratio = bonding_links / total_links if total_links > 0 else 0
        
        print(f"  Total links created: {total_links}")
        print(f"  Actual avg degree: {actual_avg_degree:.2f} (target: {self.config.target_avg_degree})")
        print(f"  Bonding: {bonding_links}, Bridging: {bridging_links}")
        print(f"  Actual bonding ratio: {actual_bonding_ratio:.2%} (target: {self.config.target_bonding_ratio:.2%})")

    def _prune_to_target_degree(self):
        """Prune edges to match target average degree"""
        # Count current edges (each edge counted once)
        current_edges = []
        for agent in self.agents.values():
            for neighbor_id, link_type in agent.connections.items():
                if agent.household_id < neighbor_id:
                    current_edges.append((agent.household_id, neighbor_id, link_type))
        
        target_edges = int(self.config.target_avg_degree * self.n_households / 2)
        
        if len(current_edges) <= target_edges:
            return  # Already at or below target
        
        # Separate by link type
        bonding = [e for e in current_edges if e[2] == 1]
        bridging = [e for e in current_edges if e[2] == 2]
        
        # Calculate target counts by type
        target_bonding = int(target_edges * self.config.target_bonding_ratio)
        target_bridging = target_edges - target_bonding
        
        # Randomly sample to keep
        keep_edges = []
        if len(bonding) > target_bonding:
            indices = np.random.choice(len(bonding), target_bonding, replace=False)
            keep_edges.extend([bonding[i] for i in indices])
        else:
            keep_edges.extend(bonding)
        
        if len(bridging) > target_bridging:
            indices = np.random.choice(len(bridging), target_bridging, replace=False)
            keep_edges.extend([bridging[i] for i in indices])
        else:
            keep_edges.extend(bridging)
        
        # Rebuild all connections
        for agent in self.agents.values():
            agent.connections = {}
        
        for h1, h2, link_type in keep_edges:
            self.agents[h1].connections[h2] = link_type
            self.agents[h2].connections[h1] = link_type
        
    
    def _determine_link_type_controlled(self, agent1: HouseholdAgent, agent2: HouseholdAgent) -> int:
        """
        Determine link type with controlled bonding ratio
        """
        f1, f2 = agent1.features, agent2.features
        
        # Calculate similarity score
        demo_similarity = (
            abs(f1['income'] - f2['income']) < 0.2 and 
            abs(f1['age'] - f2['age']) < 0.3 and
            f1['community_id'] == f2['community_id']
        )
        
        # Use target bonding ratio from config
        bonding_threshold = self.config.target_bonding_ratio
        
        if demo_similarity and np.random.random() < bonding_threshold:
            return 1  # Bonding
        else:
            return 2  # Bridging
    
    def _compute_connection_probability(self, agent1: HouseholdAgent, agent2: HouseholdAgent) -> float:
        """Compute probability of forming connection between two agents"""
        f1, f2 = agent1.features, agent2.features
        
        # Geographic proximity
        geo_dist = np.sqrt((f1['latitude'] - f2['latitude'])**2 + 
                          (f1['longitude'] - f2['longitude'])**2)
        geo_prob = np.exp(-geo_dist * 50)
        
        # Demographic similarity
        demo_sim = np.exp(-((f1['income'] - f2['income'])**2 + 
                           (f1['age'] - f2['age'])**2))
        
        # Community connection
        community_bonus = 0.3 if f1['community_id'] == f2['community_id'] else 0.0
        
        return geo_prob * demo_sim + community_bonus
    
    def _save_network_state(self, t: int):
        """Save current network state"""
        edges = []
        for agent in self.agents.values():
            for neighbor_id, link_type in agent.connections.items():
                if agent.household_id < neighbor_id:  # Avoid duplicates
                    edges.append({
                        'household_id_1': agent.household_id,
                        'household_id_2': neighbor_id,
                        'timestep': t,
                        'link_type': link_type
                    })
        
        self.network_history.extend(edges)
    
    def _compute_feature_based_probabilities(self, agent: HouseholdAgent, t: int, decision_type: str) -> Tuple[float, float]:
        """Compute base probabilities from features"""
        features = agent.features
        
        # Self-activation probability
        if decision_type == 'repair':
            if features['damage_level'] == 0:
                base_p_self = 0.0
            else:
                # Removed old time penalty: -3.0 * t / self.n_timesteps
                # Now time effect is handled separately via _compute_time_effect
                linear_self = (
                    0.8 * features['damage_level'] +
                    0.4 * features['building_value'] +
                    0.2 * features['income'] +
                    -0.2 * features['age']
                ) - 2.5
                base_p_self = 1 / (1 + np.exp(-linear_self))
                
        elif decision_type == 'vacant':
            # Calculate community damage
            community_damage = np.mean([
                self.agents[hid].features['damage_level'] 
                for hid in self.household_ids 
                if self.agents[hid].features['community_id'] == features['community_id']
            ])
            
            # Removed old time penalty: -3.0 * t / self.n_timesteps
            linear_self = (
                1.5 * community_damage +
                -0.6 * features['building_value'] +
                -0.4 * features['income']
            ) - 2.5
            base_p_self = 1 / (1 + np.exp(-linear_self))
            
        elif decision_type == 'sell':
            # Removed old time penalty: -3.0 * t / self.n_timesteps
            linear_self = (
                0.7 * features['damage_level'] +
                -0.2 * features['building_value'] +
                -0.1 * features['income']
            ) - 2.5
            base_p_self = 1 / (1 + np.exp(-linear_self))
        else:
            base_p_self = 0.001
        
        # Social responsiveness base probability
        # Removed old time penalty: -2.0 * t / self.n_timesteps
        linear_social = (
            -0.3 * features['age'] +
            0.2 * features['income'] +
            0.1 * (1 - features['race'])
        ) + 0.3
        
        base_p_social = 1 / (1 + np.exp(-linear_social))
        
        return base_p_self, base_p_social
        
    def _apply_role_adjustments(self, agent: HouseholdAgent, base_p_self: float, base_p_social: float) -> Tuple[float, float]:
        """Apply role-based adjustments to base probabilities"""
        # adjusted_p_self = np.clip(base_p_self * agent.role_params['self_multiplier'], 0.001, 0.4)
        # adjusted_p_social = np.clip(base_p_social * agent.role_params['social_multiplier'], 0.001, 0.6)
        adjusted_p_self = base_p_self * agent.role_params['self_multiplier']
        adjusted_p_social = base_p_social * agent.role_params['social_multiplier']
        if adjusted_p_self < 0.10:  # 低于8%直接清零
            adjusted_p_self = 0.0
        else:
            adjusted_p_self = min(adjusted_p_self, 0.65)  # Cap at 65%
        
        if adjusted_p_social < 0.10:  # 低于5%直接清零
            adjusted_p_social = 0.0
        else:
            adjusted_p_social = min(adjusted_p_social, 0.50)  # Cap at 70%
        
        return adjusted_p_self, adjusted_p_social
    
    def _adjust_for_dependencies(self, 
                                 p_self: float, 
                                 p_social: float,
                                 decision_type: str,
                                 current_decisions: dict) -> Tuple[float, float]:
        """
        Adjust probabilities based on decisions already made in current timestep
        
        Args:
            p_self: Base self-activation probability
            p_social: Base social influence probability
            decision_type: Current decision being evaluated ('repair', 'vacant', 'sell')
            current_decisions: Dict of decisions already made this timestep
                              e.g., {'repair': 1, 'sell': 0}
        
        Returns:
            Tuple of adjusted (p_self, p_social)
        """
        adjusted_p_self = p_self
        adjusted_p_social = p_social
        
        # Apply adjustment factors based on already-activated decisions
        for prev_decision, prev_state in current_decisions.items():
            if prev_state == 1 and prev_decision != decision_type:
                # Get adjustment factor from config
                factor = self.config.get_adjustment_factor(prev_decision, decision_type)
                
                # Apply to both self and social components
                adjusted_p_self *= factor
                adjusted_p_social *= factor
        
        return adjusted_p_self, adjusted_p_social
    
    def _compute_social_influence(self, agent: HouseholdAgent, base_p_social: float, t: int, decision_type: str) -> float:
        """Compute actual social influence from active neighbors"""
        active_neighbors = []

        # if agent.household_id == 9 and t == 5 and decision_type == 'vacant':
        #     print(f"\nAgent 9 connections: {dict(agent.connections)}")
        #     print(f"Number of connections: {len(agent.connections)}")

        
        for neighbor_id, link_type in agent.connections.items():
            if link_type > 0:
                neighbor = self.agents[neighbor_id]
                if len(neighbor.decision_history[decision_type]) > (t-1) and neighbor.decision_history[decision_type][t-1] == 1:
                    influence_strength = self._compute_pairwise_influence(
                        agent, neighbor, link_type, decision_type)
                    # print(f"  Neighbor {neighbor_id} (link_type={link_type}) influenced with strength {influence_strength:.3f}")
                    active_neighbors.append(influence_strength * base_p_social)
                    # print(f"    Base social prob: {base_p_social:.3f}, Adjusted influence: {influence_strength * base_p_social:.3f}")
        
        p_social = 1 - np.prod([1 - inf for inf in active_neighbors])
        if t == 10 and decision_type == 'repair':  # 只在t=10记录repair决策
            print(f"Agent {agent.household_id} ({agent.agent_type}): "
                f"{len(active_neighbors)} active neighbors, "
                f"p_social = {p_social:.3f}")
        if not active_neighbors:
            return 0.0
        
        return p_social
    
    # def _compute_pairwise_influence(self, agent, neighbor, link_type, decision_type):
    #     f1, f2 = agent.features, neighbor.features
        
    #     # Demographic similarity (0-1, higher = more similar)
    #     demo_sim = np.exp(-((f1['income'] - f2['income'])**2 + 
    #                     (f1['age'] - f2['age'])**2 + 
    #                     (f1['race'] - f2['race'])**2))
    #     demo_dist = np.sqrt(demo_sim / 3)
    #     demo_sim = np.exp(-demo_dist * 2) 
        
    #     # Geographic distance (normalized)
    #     geo_dist = np.sqrt((f1['latitude'] - f2['latitude'])**2 + 
    #                     (f1['longitude'] - f2['longitude'])**2)
        
    #     # Normalize geo_dist to 0-1 range (max possible distance in your data ≈ 0.28)
    #     # Use a gentler decay: distance within 0.1 degrees → high influence
    #     geo_sim = np.exp(-geo_dist * 5)  # Changed from 100 to 5
        
    #     # Link type multiplier
    #     link_multiplier = 2.0 if link_type == 1 else 1.0  # Reduced bridging from 1.5 to 1.0
        
    #     # Combine factors with higher base weight
    #     influence = demo_sim * geo_sim * link_multiplier * 1.2  # Changed from 0.3 to 0.8
        
    #     return influence

    def _compute_pairwise_influence(self, agent, neighbor, link_type, decision_type):
        f1, f2 = agent.features, neighbor.features
        
        # 1. Demographic similarity - 线性相似度（避免指数衰减）
        income_sim = 1 - abs(f1['income'] - f2['income'])
        age_sim = 1 - abs(f1['age'] - f2['age'])
        race_sim = 1 - abs(f1['race'] - f2['race']) / 1.0
        demo_sim = (income_sim + age_sim + race_sim) / 3  # 平均：0-1
        
        # 2. Geographic proximity - 分段（避免过度衰减）
        geo_dist = np.sqrt((f1['latitude'] - f2['latitude'])**2 + 
                        (f1['longitude'] - f2['longitude'])**2)
        
        if geo_dist < 0.05:      # 很近
            geo_prox = 0.9
        elif geo_dist < 0.15:    # 中等
            geo_prox = 0.6
        elif geo_dist < 0.25:    # 较远
            geo_prox = 0.3
        else:                     # 很远
            geo_prox = 0.1
        
        # 3. Link type effect - 增强base influence
        if link_type == 1:  # Bonding
            base_influence = 0.2  # 基础35%
            influence = base_influence + 0.3 * demo_sim + 0.15 * geo_prox
        else:  # Bridging
            base_influence = 0.05  # 基础20%
            influence = base_influence + 0.15 * demo_sim + 0.1 * geo_prox
        if influence < 0.05:  # 低于15%直接清零
            influence = 0.0
        
        return min(influence, 0.5)  # Cap at 85%
    
    def _make_decision(self, agent: HouseholdAgent, t: int, decision_type: str, 
                    p_self: float, p_social: float, current_decisions: dict) -> int:
        """
        Make decision for agent based on probabilities, with time gating and dependency adjustments
        
        Args:
            agent: The household agent making the decision
            t: Current timestep
            decision_type: Type of decision ('repair', 'vacant', 'sell')
            p_self: Base self-activation probability
            p_social: Base social influence probability
            current_decisions: Decisions already made this timestep
            
        Returns:
            Binary decision (0 or 1)
        """
        # Check if already activated in history (irreversibility)
        if len(agent.decision_history[decision_type]) > t and agent.decision_history[decision_type][t] == 1:
            return 1
        
        # # Check if other decisions already active (mutual exclusivity check for historical states)
        # other_decisions = [d for d in ['repair', 'vacant', 'sell'] if d != decision_type]
        # if any(len(agent.decision_history[d]) > t and agent.decision_history[d][t] == 1 for d in other_decisions):
        #     return 0
        
        # NEW: Apply time gating - check if decision is active at this time
        time_effect = self._compute_time_effect(decision_type, t)
        
        # NEW: Apply dependency adjustments based on decisions made this timestep
        p_self_adjusted, p_social_adjusted = self._adjust_for_dependencies(
            p_self, p_social, decision_type, current_decisions
        )
        
        # NEW: Apply time effect to adjusted probabilities
        p_self_final = p_self_adjusted * time_effect
        p_social_final = p_social_adjusted * time_effect
        
        # Combine probabilities
        final_prob = 1 - (1 - p_self_final) * (1 - p_social_final)
        
        # Role-specific caps (existing logic)
        if agent.agent_type == 'isolated':
            final_prob = min(final_prob, 0.12) 
        elif agent.agent_type == 'resistant':
            final_prob = min(final_prob, 0.08)  
        elif agent.agent_type == 'social_follower':
            final_prob = min(final_prob, 0.35)  
        else:  # early_adopter
            final_prob = min(final_prob, 0.45)  
        
        # Sample decision
        decision = int(np.random.rand() < final_prob)
        
        # Log probability details
        self._log_probability_details(agent, t, decision_type, p_self_final, p_social_final, final_prob, decision)
        
        return decision
    
    def _log_probability_details(self, agent: HouseholdAgent, t: int, decision_type: str, 
                               p_self: float, p_social: float, final_prob: float, decision: int):
        """Log detailed probability information"""
        active_neighbors = []
        for neighbor_id, link_type in agent.connections.items():
            if link_type > 0:
                neighbor = self.agents[neighbor_id]
                if (len(neighbor.decision_history[decision_type]) > t and 
                    neighbor.decision_history[decision_type][t] == 1):
                    active_neighbors.append({
                        'neighbor_id': neighbor_id,
                        'link_type': link_type,
                        'influence_prob': self._compute_pairwise_influence(agent, neighbor, link_type, decision_type)
                    })
        
        self.probability_log.append({
            'timestep': t,
            'household_id': agent.household_id,
            'agent_type': agent.agent_type,
            'decision_type': decision_type,
            'self_activation_prob': p_self,
            'social_influence_prob': p_social,
            'final_activation_prob': final_prob,
            'active_neighbors': len(active_neighbors),
            'neighbor_details': active_neighbors,
            'actual_decision': decision,
            'features': agent.features.copy()
        })
    
    def _update_network(self, t: int):
        # At t=0, no previous vacant decisions exist, just save initial network
        if t == 0:
            self._save_network_state(t)
            return
        
        """Update network structure"""
        # Remove bridging links when someone becomes vacant
        for agent in self.agents.values():
            if (len(agent.decision_history['vacant']) > t and 
                agent.decision_history['vacant'][t-1] == 1):
                to_remove = []
                for neighbor_id, link_type in agent.connections.items():
                    if link_type == 2:
                        to_remove.append(neighbor_id)
                        if neighbor_id in self.agents and agent.household_id in self.agents[neighbor_id].connections:
                            del self.agents[neighbor_id].connections[agent.household_id]
                
                for neighbor_id in to_remove:
                    if neighbor_id in agent.connections:
                        del agent.connections[neighbor_id]
        
        # self._form_new_connections(t)
        self._save_network_state(t)
    
    def _form_new_connections(self, t: int):
        """Form new connections based on agent behavior"""
        for agent in self.agents.values():
            connection_prob = agent.role_params['network_tendency'] * 0.05
            if np.random.random() < connection_prob:
                candidates = []
                for other_id, other_agent in self.agents.items():
                    if (other_id != agent.household_id and 
                        other_id not in agent.connections and
                        not (len(other_agent.decision_history['vacant']) > t and 
                             other_agent.decision_history['vacant'][t] == 1)):
                        
                        connection_prob_val = self._compute_connection_probability(agent, other_agent)
                        candidates.append((other_id, connection_prob_val))
                
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = candidates[:min(8, len(candidates))]
                    if top_candidates[0][1] > 0.05:
                        new_neighbor_id = np.random.choice([c[0] for c in top_candidates])
                        link_type = self._determine_link_type_controlled(agent, self.agents[new_neighbor_id])
                        
                        agent.connections[new_neighbor_id] = link_type
                        self.agents[new_neighbor_id].connections[agent.household_id] = link_type
    
    def simulate(self) -> Dict:
        """Main simulation loop"""
        print(f"\n{'='*70}")
        print(f"Starting ABM simulation: {self.config.name}")
        print(f"{'='*70}")
        
        # Initialize network and decisions
        self._initialize_network()
        self._prune_to_target_degree()  
        self._save_network_state(0) 
        self.generate_initial_decisions_controlled()

        role_activation_log = []
        
        print(f"\nSimulating {self.n_timesteps} timesteps...")
        
        for t in tqdm(range(1, self.n_timesteps), desc="Simulating"):
            # Initialize decision arrays
            for agent in self.agents.values():
                for decision_type in ['repair', 'vacant', 'sell']:
                    while len(agent.decision_history[decision_type]) <= t:
                        agent.decision_history[decision_type].append(
                            agent.decision_history[decision_type][-1])
                                
            # Debug: Count activations by role for this timestep
            step_stats = {'timestep': t}
            for role in ['early_adopter', 'social_follower', 'resistant', 'isolated']:
                agents_of_role = [a for a in self.agents.values() if a.agent_type == role]
                
                for decision_type in ['repair', 'vacant', 'sell']:
                    newly_activated = sum(
                        1 for a in agents_of_role 
                        if len(a.decision_history[decision_type]) > t and
                        a.decision_history[decision_type][t] == 1 and
                        a.decision_history[decision_type][t-1] == 0
                    )
                    step_stats[f'{role}_{decision_type}_new'] = newly_activated
                    
                    # Also track average probabilities
                    probs = [
                        log['final_activation_prob'] 
                        for log in self.probability_log 
                        if log['timestep'] == t-1 and 
                        log['agent_type'] == role and 
                        log['decision_type'] == decision_type
                    ]
                    step_stats[f'{role}_{decision_type}_avg_prob'] = np.mean(probs) if probs else 0
            
            role_activation_log.append(step_stats)
            
            # NEW: Sequential decision making with dependency tracking
            for agent in self.agents.values():
                # Track decisions made in THIS timestep for THIS agent
                current_decisions = {}
                
                # Get decision order (repair -> sell -> vacant)
                decision_order = self._get_decision_order()
                
                for decision_type in decision_order:
                    # Get historical state from previous timestep
                    prev_state = agent.decision_history[decision_type][t-1] if t > 0 else agent.decision_history[decision_type][0]
                    
                    if prev_state == 1:
                        # Already active (irreversibility)
                        agent.decision_history[decision_type][t] = 1
                        current_decisions[decision_type] = 1
                        continue
                                    
                    # Compute base probabilities
                    base_p_self, base_p_social = self._compute_feature_based_probabilities(
                        agent, t, decision_type
                    )
                    
                    # Apply role adjustments
                    adj_p_self, adj_p_social = self._apply_role_adjustments(
                        agent, base_p_self, base_p_social
                    )
                    
                    # Compute social influence
                    social_influence = self._compute_social_influence(
                        agent, adj_p_social, t, decision_type
                    )
                    
                    # Log p_self and p_ji values
                    self.p_self_all_values.append({
                        'time_step': t,
                        'dimension': ['vacant', 'repair', 'sell'].index(decision_type),
                        'dimension_name': decision_type,
                        'household_id': agent.household_id,
                        'p_self_value': adj_p_self
                    })

                    for neighbor_id, link_type in agent.connections.items():
                        if link_type > 0:
                            neighbor = self.agents[neighbor_id]
                            if (len(neighbor.decision_history[decision_type]) > t and 
                                neighbor.decision_history[decision_type][t] == 1):
                                pairwise_influence = self._compute_pairwise_influence(
                                    agent, neighbor, link_type, decision_type
                                ) * adj_p_social
                            else:
                                pairwise_influence = 0.0
                            
                            self.p_ji_all_values.append({
                                'time_step': t,
                                'dimension': ['vacant', 'repair', 'sell'].index(decision_type),
                                'dimension_name': decision_type,
                                'household_i': neighbor_id,
                                'household_j': agent.household_id,
                                'p_ji_value': pairwise_influence
                            })
                    
                    # NEW: Make decision with current_decisions context
                    decision = self._make_decision(
                        agent, t, decision_type, adj_p_self, social_influence, current_decisions
                    )
                    
                    # Update both history and current tracking
                    agent.decision_history[decision_type][t] = decision
                    current_decisions[decision_type] = decision

            # Update network
            self._update_network(t)

            # import json
            # with open(f"{self.config.name}_debug_log.json", 'w') as f:
            #     json.dump(role_activation_log, f, indent=2)
            
            # Print summary
            print("\n=== Role-based Activation Summary ===")
            for role in ['early_adopter', 'social_follower', 'resistant', 'isolated']:
                agents_of_role = [a for a in self.agents.values() if a.agent_type == role]
                n_role = len(agents_of_role)
                
                for decision_type in ['repair', 'vacant', 'sell']:
                    final_activated = sum(
                        1 for a in agents_of_role 
                        if a.decision_history[decision_type][-1] == 1
                    )
                    print(f"{role} - {decision_type}: {final_activated}/{n_role} ({final_activated/n_role*100:.1f}%)")     
        
        print("\nSimulation completed!")
        return self._prepare_output_data()
    
    def _prepare_output_data(self) -> Dict:
        """Prepare data in required format (matching original structure)"""
        # 1. Household states
        states_data = []
        for agent in self.agents.values():
            for t in range(self.n_timesteps):
                states_data.append({
                    'household_id': agent.household_id,
                    'timestep': t,
                    'repair': agent.decision_history['repair'][t] if len(agent.decision_history['repair']) > t else 0,
                    'vacant': agent.decision_history['vacant'][t] if len(agent.decision_history['vacant']) > t else 0,
                    'sell': agent.decision_history['sell'][t] if len(agent.decision_history['sell']) > t else 0
                })
        
        household_states = pd.DataFrame(states_data)
        
        # 2. Ground truth network
        ground_truth_network = pd.DataFrame(self.network_history)
        
        # 3. Observed network with TWO missing mechanisms
        observed_network = self._apply_missing_mechanisms(ground_truth_network)
        
        # 4. Household features (with proper column ordering)
        features_data = []
        for agent in self.agents.values():
            features = {
                'household_id': agent.household_id,
                'building_value': agent.features['building_value'],
                'income': agent.features['income'], 
                'damage_level': agent.features['damage_level'],
                'population_scaled': agent.features['population_scaled'],
                'age': agent.features['age'],
                'race': agent.features['race'],
                'community_id': agent.features['community_id'],
                'latitude': agent.features['latitude'],
                'longitude': agent.features['longitude'],
                'agent_type': agent.agent_type
            }
            features_data.append(features)
        
        household_features = pd.DataFrame(features_data)
        
        # One-hot encode community_id
        community_dummies = pd.get_dummies(household_features['community_id'], prefix='community')
        
        # Reorder columns
        feature_columns = ['household_id', 'building_value', 'income', 'damage_level', 
                          'population_scaled', 'age', 'race', 'latitude', 'longitude', 'agent_type']
        
        household_features_final = household_features[feature_columns].copy()
        household_features_final = pd.concat([household_features_final, community_dummies], axis=1)
        
        # 5. Household locations
        household_locations = pd.DataFrame([
            {
                'household_id': agent.household_id,
                'latitude': agent.features['latitude'],
                'longitude': agent.features['longitude']
            }
            for agent in self.agents.values()
        ])
        
        # Calculate missing statistics
        missing_stats = self._calculate_missing_statistics(ground_truth_network, observed_network)
        
        return {
            'household_states': household_states,
            'ground_truth_network': ground_truth_network,
            'observed_network': observed_network,
            'household_features': household_features_final,
            'household_locations': household_locations,
            'probability_log': self.probability_log,
            'p_self_all_values': self.p_self_all_values,  # NEW
            'p_ji_all_values': self.p_ji_all_values,      # NEW
            'agent_summary': self._get_agent_summary(),
            'missing_statistics': missing_stats
        }
    
    def _apply_missing_mechanisms(self, ground_truth_network: pd.DataFrame) -> pd.DataFrame:
        """
        Apply two missing mechanisms to create observed network:
        1. Structural missing: edges permanently unobservable (across all time steps)
        2. Temporal missing: random time point records missing (only for observable edges)
        """
        observed_network = ground_truth_network.copy()
        
        # ========== MECHANISM 1: Structural Missing ==========
        # Identify unique edges (ignoring time dimension)
        unique_edges = ground_truth_network[['household_id_1', 'household_id_2', 'link_type']].drop_duplicates()
        
        # Determine which edges are structurally observable
        edge_observable = {}  # Key: (id1, id2), Value: True/False
        
        print(f"\nApplying Missing Mechanisms:")
        print(f"  Ground truth edges: {len(unique_edges)} unique edges")
        
        for _, edge in unique_edges.iterrows():
            id1, id2, link_type = edge['household_id_1'], edge['household_id_2'], edge['link_type']
            # Create canonical edge key (smaller id first)
            edge_key = (min(id1, id2), max(id1, id2))
            
            # Decide if this edge is permanently observable based on link type
            if link_type == 1:  # Bonding link
                is_observable = np.random.random() > self.config.rho_structural_bonding
            else:  # Bridging link (type 2)
                is_observable = np.random.random() > self.config.rho_structural_bridging
            
            edge_observable[edge_key] = is_observable
        
        # Count structurally missing edges
        structural_observable_count = sum(edge_observable.values())
        structural_missing_count = len(unique_edges) - structural_observable_count
        
        print(f"  Structural missing applied:")
        print(f"    - Observable edges: {structural_observable_count}")
        print(f"    - Structurally missing edges: {structural_missing_count} ({structural_missing_count/len(unique_edges)*100:.1f}%)")
        
        # Remove all time-step records of structurally missing edges
        structural_mask = observed_network.apply(
            lambda row: edge_observable.get(
                (min(row['household_id_1'], row['household_id_2']),
                 max(row['household_id_1'], row['household_id_2'])),
                True  # Default to observable if not found
            ),
            axis=1
        )
        observed_network = observed_network[structural_mask].reset_index(drop=True)
        
        print(f"    - Records after structural missing: {len(observed_network)} (from {len(ground_truth_network)})")
        
        # ========== MECHANISM 2: Temporal Missing ==========
        # Apply only to remaining observable edges
        temporal_mask = np.ones(len(observed_network), dtype=bool)
        
        for idx, row in observed_network.iterrows():
            # Apply temporal missing based on link type
            if row['link_type'] == 1:  # Bonding link
                if np.random.random() < self.config.rho_temporal_bonding:
                    temporal_mask[idx] = False
            else:  # Bridging link (type 2)
                if np.random.random() < self.config.rho_temporal_bridging:
                    temporal_mask[idx] = False
        
        temporal_missing_count = len(observed_network) - temporal_mask.sum()
        print(f"  Temporal missing applied:")
        print(f"    - Records removed: {temporal_missing_count} ({temporal_missing_count/len(observed_network)*100:.1f}%)")
        
        observed_network = observed_network[temporal_mask].reset_index(drop=True)
        
        print(f"  Final observed network: {len(observed_network)} records (from {len(ground_truth_network)} ground truth)")
        print(f"  Overall missing rate: {(1 - len(observed_network)/len(ground_truth_network))*100:.1f}%")
        
        return observed_network
    
    def _calculate_missing_statistics(self, ground_truth: pd.DataFrame, observed: pd.DataFrame) -> Dict:
        """Calculate detailed statistics about missing mechanisms"""
        gt_total = len(ground_truth)
        obs_total = len(observed)
        
        # By link type
        gt_bonding = len(ground_truth[ground_truth['link_type'] == 1])
        gt_bridging = len(ground_truth[ground_truth['link_type'] == 2])
        obs_bonding = len(observed[observed['link_type'] == 1])
        obs_bridging = len(observed[observed['link_type'] == 2])
        
        # Unique edges
        gt_unique = ground_truth[['household_id_1', 'household_id_2']].drop_duplicates()
        obs_unique = observed[['household_id_1', 'household_id_2']].drop_duplicates()
        
        stats = {
            'ground_truth_total_records': gt_total,
            'observed_total_records': obs_total,
            'overall_missing_rate': round((1 - obs_total / gt_total) * 100, 2) if gt_total > 0 else 0,
            
            'ground_truth_bonding_records': gt_bonding,
            'ground_truth_bridging_records': gt_bridging,
            'observed_bonding_records': obs_bonding,
            'observed_bridging_records': obs_bridging,
            
            'bonding_missing_rate': round((1 - obs_bonding / gt_bonding) * 100, 2) if gt_bonding > 0 else 0,
            'bridging_missing_rate': round((1 - obs_bridging / gt_bridging) * 100, 2) if gt_bridging > 0 else 0,
            
            'ground_truth_unique_edges': len(gt_unique),
            'observed_unique_edges': len(obs_unique),
            'unique_edges_missing_rate': round((1 - len(obs_unique) / len(gt_unique)) * 100, 2) if len(gt_unique) > 0 else 0,
            
            'config_structural_bonding': self.config.rho_structural_bonding,
            'config_structural_bridging': self.config.rho_structural_bridging,
            'config_temporal_bonding': self.config.rho_temporal_bonding,
            'config_temporal_bridging': self.config.rho_temporal_bridging
        }
        
        return stats
    
    def _get_agent_summary(self) -> Dict:
        """Get summary of agent types and their behavior"""
        summary = {}
        for agent_type in self.agent_types.keys():
            agents_of_type = [a for a in self.agents.values() if a.agent_type == agent_type]
            if agents_of_type:
                avg_connections = np.mean([len(a.connections) for a in agents_of_type])
                total_repair = sum(a.decision_history['repair'][-1] if a.decision_history['repair'] else 0 for a in agents_of_type)
                total_vacant = sum(a.decision_history['vacant'][-1] if a.decision_history['vacant'] else 0 for a in agents_of_type)
                total_sell = sum(a.decision_history['sell'][-1] if a.decision_history['sell'] else 0 for a in agents_of_type)
            else:
                avg_connections = 0
                total_repair = total_vacant = total_sell = 0
                
            summary[agent_type] = {
                'count': len(agents_of_type),
                'avg_connections': avg_connections,
                'total_decisions': {
                    'repair': total_repair,
                    'vacant': total_vacant,
                    'sell': total_sell
                }
            }
        return summary


def save_abm_data(data: Dict, output_dir: str, config: ABMScenarioConfig):
    """
    Save generated ABM data to files (matching original structure)
    
    Args:
        data: Dictionary with all generated data
        output_dir: Output directory path
        config: Configuration used for generation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files (matching original names)
    data['household_states'].to_csv(f"{output_dir}/household_states.csv", index=False)
    data['ground_truth_network'].to_csv(f"{output_dir}/ground_truth_network.csv", index=False)
    data['observed_network'].to_csv(f"{output_dir}/observed_network.csv", index=False)
    data['household_features'].to_csv(f"{output_dir}/household_features.csv", index=False)
    data['household_locations'].to_csv(f"{output_dir}/household_locations.csv", index=False)
    p_self_all_df = pd.DataFrame(data['p_self_all_values'])
    p_ji_all_df = pd.DataFrame(data['p_ji_all_values'])

    p_self_all_df.to_csv(f"{output_dir}/p_self_all_values.csv", index=False)
    p_ji_all_df.to_csv(f"{output_dir}/p_ji_all_values.csv", index=False)
    
    # Save probability log and summary
    with open(f"{output_dir}/detailed_probabilities.pkl", 'wb') as f:
        pickle.dump({
            'probability_log': data['probability_log'],
            'agent_summary': data['agent_summary'],
            'missing_statistics': data['missing_statistics'],
            'metadata': {
                'n_households': len(data['household_features']),
                'n_timesteps': data['household_states']['timestep'].max() + 1,
                'generation_method': 'ABM_configurable',
                'scenario_name': config.name
            }
        }, f)
    
    # Compute and save similarity matrix
    features_matrix = data['household_features'][['income', 'age', 'race', 'building_value']].values
    from scipy.spatial.distance import pdist, squareform
    similarity_matrix = 1 - squareform(pdist(features_matrix, metric='euclidean'))
    similarity_df = pd.DataFrame(similarity_matrix, 
                                index=data['household_features']['household_id'],
                                columns=data['household_features']['household_id'])
    similarity_df.to_csv(f"{output_dir}/similarity_matrix.csv")
    
    # Save configuration
    from abm_config import save_config_to_json
    save_config_to_json(config, f"{output_dir}/config.json")
    
    print(f"\nData saved to {output_dir}/")
    print(f"Files generated:")
    print(f"  - household_states.csv: {len(data['household_states'])} rows")
    print(f"  - ground_truth_network.csv: {len(data['ground_truth_network'])} rows")
    print(f"  - observed_network.csv: {len(data['observed_network'])} rows")
    print(f"  - household_features.csv: {len(data['household_features'])} rows")
    print(f"  - household_locations.csv: {len(data['household_locations'])} rows")
    print(f"  - similarity_matrix.csv: {similarity_df.shape[0]}x{similarity_df.shape[1]} matrix")
    print(f"  - config.json: Scenario configuration")
    print(f"  - detailed_probabilities.pkl: {len(data['probability_log'])} records")
    print(f"  - p_self_all_values.csv: {len(p_self_all_df)} rows")
    print(f"  - p_ji_all_values.csv: {len(p_ji_all_df)} rows")

    
    # Display missing mechanisms summary
    print(f"\nMissing Mechanisms Applied:")
    ms = data['missing_statistics']
    print(f"  Ground truth: {ms['ground_truth_unique_edges']} unique edges, {ms['ground_truth_total_records']} total records")
    print(f"  Observed: {ms['observed_unique_edges']} unique edges, {ms['observed_total_records']} total records")
    print(f"  Overall missing rate: {ms['overall_missing_rate']}%")
    print(f"  Structural missing (bonding): {ms['config_structural_bonding']*100}%")
    print(f"  Structural missing (bridging): {ms['config_structural_bridging']*100}%")
    print(f"  Temporal missing (bonding): {ms['config_temporal_bonding']*100}%")
    print(f"  Temporal missing (bridging): {ms['config_temporal_bridging']*100}%")

