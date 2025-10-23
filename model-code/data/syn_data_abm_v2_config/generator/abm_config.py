
# abm_config.py
from dataclasses import dataclass
import json
from typing import Dict, Any, List

@dataclass
class ABMScenarioConfig:
    """Configuration for a single ABM scenario"""
    # Basic scenario information
    name: str
    description: str
    
    # Agent role distribution (must sum to 1.0)
    early_adopter_ratio: float = 0.15
    social_follower_ratio: float = 0.25
    resistant_ratio: float = 0.45
    isolated_ratio: float = 0.15
    
    # Network density parameters
    target_avg_degree: float = 3.0  # Target average number of connections per agent
    target_bonding_ratio: float = 0.2  # Target ratio of bonding links (20% bonding, 80% bridging)
    
    # Decision seeding parameters
    target_seed_ratio: float = 0.10  # Total proportion of households with initial decisions (10%)
    repair_seed_ratio: float = 0.6  # Repair takes 60% of seeds
    vacant_seed_ratio: float = 0.3  # Vacant takes 30% of seeds
    sell_seed_ratio: float = 0.1   # Sell takes 10% of seeds
    
    # Simulation parameters
    n_households: int = 200
    n_timesteps: int = 24
    random_seed: int = 42
    
    # Missing mechanisms for observed network
    # Structural missing: edges that are permanently unobservable across all time steps
    rho_structural_bonding: float = 0.5    # 50% of bonding edges are permanently unobservable
    rho_structural_bridging: float = 0.5   # 50% of bridging edges are permanently unobservable
    
    # Temporal missing: random time point records missing (applied only to structurally observable edges)
    rho_temporal_bonding: float = 0.2      # 20% of observable bonding records randomly missing at specific time points
    rho_temporal_bridging: float = 0.2     # 20% of observable bridging records randomly missing at specific time points

        # ========== NEW: Decision timing control ==========
    # When each decision type can start being activated
    repair_start_time: int = 0      # Repair starts immediately after disaster
    vacant_start_time: int = 2      # Vacant starts at month 3
    sell_start_time: int = 4        # Sell starts at month 6
    
    # Time dynamics: peak time for each decision (for bell-curve shape)
    repair_peak_time: int = 4       # Repair peaks at month 4
    vacant_peak_time: int = 8      # Vacant peaks at month 10
    sell_peak_time: int = 10        # Sell peaks at month 14
    
    # ========== NEW: Decision interaction adjustments ==========
    # Adjustment factors: if decision X is active, how does it affect decision Y's probability?
    # Factor < 1.0 means suppression, > 1.0 means promotion
    
    # Repair's effect on others
    repair_to_vacant_factor: float = 0.8    # Repair slightly suppresses vacant
    repair_to_sell_factor: float = 0.5      # Repair moderately suppresses sell
    
    # Vacant's effect on others  
    vacant_to_repair_factor: float = 0.8    # Vacant strongly suppresses repair
    vacant_to_sell_factor: float = 1.0      # Vacant slightly promotes sell
    
    # Sell's effect on others
    sell_to_repair_factor: float = 0.01      # Sell strongly suppresses repair
    sell_to_vacant_factor: float = 1.5      # Sell promotes vacant
    
    def get_adjustment_factor(self, from_decision: str, to_decision: str) -> float:
        """Get adjustment factor from one decision to another"""
        key = f"{from_decision}_to_{to_decision}_factor"
        return getattr(self, key, 1.0)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Check role ratios sum to 1.0
        total_ratio = (self.early_adopter_ratio + self.social_follower_ratio + 
                      self.resistant_ratio + self.isolated_ratio)
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Role ratios must sum to 1.0, got {total_ratio}")
        
        # Check seed ratios sum to 1.0
        total_seed = self.repair_seed_ratio + self.vacant_seed_ratio + self.sell_seed_ratio
        if not (0.99 <= total_seed <= 1.01):
            raise ValueError(f"Seed ratios must sum to 1.0, got {total_seed}")
        
        # Check all ratios are positive
        if any(r < 0 for r in [self.early_adopter_ratio, self.social_follower_ratio,
                                self.resistant_ratio, self.isolated_ratio,
                                self.target_seed_ratio, self.repair_seed_ratio,
                                self.vacant_seed_ratio, self.sell_seed_ratio,
                                self.target_bonding_ratio, self.rho_structural_bonding,
                                self.rho_structural_bridging, self.rho_temporal_bonding,
                                self.rho_temporal_bridging]):
            raise ValueError("All ratio parameters must be non-negative")
        
        # Check missing rates are between 0 and 1
        if not all(0 <= r <= 1 for r in [self.rho_structural_bonding, self.rho_structural_bridging,
                                          self.rho_temporal_bonding, self.rho_temporal_bridging]):
            raise ValueError("All missing rate parameters must be between 0 and 1")



def get_abm_scenario(name: str) -> ABMScenarioConfig:
    """Get ABM scenario configuration by name"""
    for scenario in ALL_ABM_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Scenario '{name}' not found. Available: {[s.name for s in ALL_ABM_SCENARIOS]}")

def save_config_to_json(config: ABMScenarioConfig, filepath: str):
    """Save configuration to JSON file"""
    config_dict = {
        'name': config.name,
        'description': config.description,
        'role_distribution': {
            'early_adopter_ratio': config.early_adopter_ratio,
            'social_follower_ratio': config.social_follower_ratio,
            'resistant_ratio': config.resistant_ratio,
            'isolated_ratio': config.isolated_ratio
        },
        'network_density': {
            'target_avg_degree': config.target_avg_degree,
            'target_bonding_ratio': config.target_bonding_ratio
        },
        'decision_seeding': {
            'target_seed_ratio': config.target_seed_ratio,
            'repair_seed_ratio': config.repair_seed_ratio,
            'vacant_seed_ratio': config.vacant_seed_ratio,
            'sell_seed_ratio': config.sell_seed_ratio
        },
        'simulation_parameters': {
            'n_households': config.n_households,
            'n_timesteps': config.n_timesteps,
            'random_seed': config.random_seed
        },
        'missing_mechanisms': {
            'structural_missing': {
                'rho_bonding': config.rho_structural_bonding,
                'rho_bridging': config.rho_structural_bridging,
                'description': 'Edges permanently unobservable across all time steps'
            },
            'temporal_missing': {
                'rho_bonding': config.rho_temporal_bonding,
                'rho_bridging': config.rho_temporal_bridging,
                'description': 'Random time point records missing for observable edges'
            }
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)



# ==================== Predefined Scenarios ====================

ABM_SPARSE_LOW_A = ABMScenarioConfig(
    name="ABM_Sparse_LowSeed_A",
    description="Sparse network with low seeding - Only temporal missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=5.0,
    target_bonding_ratio=0.15,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=1000
)

ABM_SPARSE_LOW_B = ABMScenarioConfig(
    name="ABM_Sparse_LowSeed_B",
    description="Sparse network with low seeding - Only structural missing",
   early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=5.0,
    target_bonding_ratio=0.15,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Only structural missing
    rho_structural_bonding=0.5,
    rho_structural_bridging=0.5,
    rho_temporal_bonding=0.0,
    rho_temporal_bridging=0.0,
    random_seed=2000
)

# ========== Sparse High Seed ==========
ABM_SPARSE_HIGH_A = ABMScenarioConfig(
    name="ABM_Sparse_HighSeed_A",
    description="Sparse network with high seeding - Only temporal missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=5.5,
    target_bonding_ratio=0.15,
    target_seed_ratio=0.18,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=2002
)

ABM_SPARSE_HIGH_B = ABMScenarioConfig(
    name="ABM_Sparse_HighSeed_B",
    description="Sparse network with high seeding - Only structural missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=5.5,
    target_bonding_ratio=0.15,
    target_seed_ratio=0.18,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Only structural missing
    rho_structural_bonding=0.5,
    rho_structural_bridging=0.5,
    rho_temporal_bonding=0.0,
    rho_temporal_bridging=0.0,
    random_seed=2010
)

# ========== Medium Low Seed ==========
ABM_MEDIUM_LOW_A = ABMScenarioConfig(
    name="ABM_Medium_LowSeed_A",
    description="Medium density network with low seeding - Only temporal missing",
   early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=7.5,
    target_bonding_ratio=0.2,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=2003
)

ABM_MEDIUM_LOW_B = ABMScenarioConfig(
    name="ABM_Medium_LowSeed_B",
    description="Medium density network with low seeding - Only structural missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=7.5,
    target_bonding_ratio=0.2,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Only structural missing
    rho_structural_bonding=0.5,
    rho_structural_bridging=0.5,
    rho_temporal_bonding=0.0,
    rho_temporal_bridging=0.0,
    random_seed=2013
)

# ========== Medium High Seed ==========
ABM_MEDIUM_HIGH_A = ABMScenarioConfig(
    name="ABM_Medium_HighSeed_A",
    description="Medium density network with high seeding - Only temporal missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=7.5,
    target_bonding_ratio=0.2,
    target_seed_ratio=0.18,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=2004
)

ABM_MEDIUM_HIGH_B = ABMScenarioConfig(
    name="ABM_Medium_HighSeed_B",
    description="Medium density network with high seeding - Only structural missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=7.5,
    target_bonding_ratio=0.2,
    target_seed_ratio=0.18,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Only structural missing
    rho_structural_bonding=0.5,
    rho_structural_bridging=0.5,
    rho_temporal_bonding=0.0,
    rho_temporal_bridging=0.0,
    random_seed=2014
)

# ========== Dense Low Seed ==========
ABM_DENSE_LOW_A = ABMScenarioConfig(
    name="ABM_Dense_LowSeed_A",
    description="Dense network with low seeding - Only temporal missing",
   early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=10.0,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=2005
)

ABM_DENSE_LOW_B = ABMScenarioConfig(
    name="ABM_Dense_LowSeed_B",
    description="Dense network with low seeding - Only structural missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=10.0,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Only structural missing
    rho_structural_bonding=0.5,
    rho_structural_bridging=0.5,
    rho_temporal_bonding=0.0,
    rho_temporal_bridging=0.0,
    random_seed=2015
)

# ========== Dense High Seed ==========
ABM_DENSE_HIGH_A = ABMScenarioConfig(
    name="ABM_Dense_HighSeed_A",
    description="Dense network with high seeding - Only temporal missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=10.0,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.18,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=2006
)

ABM_DENSE_HIGH_B = ABMScenarioConfig(
    name="TEST1",
    description="Dense network with high seeding - Only structural missing",
   early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=10.0,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.18,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Only structural missing
    rho_structural_bonding=0.5,
    rho_structural_bridging=0.5,
    rho_temporal_bonding=0.0,
    rho_temporal_bridging=0.0,
    random_seed=2025
)

TEST1 = ABMScenarioConfig(
    name="TEST1",
    description="Dense network with high seeding - Only temporal missing",
    early_adopter_ratio=0.10,
    social_follower_ratio=0.45,
    resistant_ratio=0.35,
    isolated_ratio=0.10,
    target_avg_degree=10.0,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.1,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=1001
)

TEST2 = ABMScenarioConfig(
    name="TEST2",
    description="Dense network with high seeding - Only temporal missing",
    early_adopter_ratio=0.05,
    social_follower_ratio=0.25,
    resistant_ratio=0.50,
    isolated_ratio=0.20,
    target_avg_degree=2.5,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.05,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=986
)

ABM_DENSE_LOW_A_n50 = ABMScenarioConfig(
    name="ABM_Dense_LowSeed_A_n50",
    description="Dense network with low seeding - Only temporal missing",
    n_households=50,
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=10.0,
    target_bonding_ratio=0.25,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=2005
)

ABM_SPARSE_LOW_A_n50 = ABMScenarioConfig(
    name="ABM_Sparse_LowSeed_A_n50",
    description="Sparse network with low seeding - Only temporal missing",
    n_households=50,
    early_adopter_ratio=0.10,
    social_follower_ratio=0.25,
    resistant_ratio=0.30,
    isolated_ratio=0.35,
    target_avg_degree=5.0,
    target_bonding_ratio=0.15,
    target_seed_ratio=0.08,
    repair_seed_ratio=0.6,
    vacant_seed_ratio=0.3,
    sell_seed_ratio=0.1,
    # Both mechanisms active
    rho_structural_bonding=0.0,
    rho_structural_bridging=0.0,
    rho_temporal_bonding=0.5,
    rho_temporal_bridging=0.5,
    random_seed=1000
)

# List of all scenarios for easy iteration
# ALL_ABM_SCENARIOS = [
#     ABM_SPARSE_LOW_A, ABM_SPARSE_LOW_B,
#     ABM_SPARSE_HIGH_A, ABM_SPARSE_HIGH_B,
#     ABM_MEDIUM_LOW_A, ABM_MEDIUM_LOW_B,
#     ABM_MEDIUM_HIGH_A, ABM_MEDIUM_HIGH_B,
#     ABM_DENSE_LOW_A, ABM_DENSE_LOW_B,
#     ABM_DENSE_HIGH_A, ABM_DENSE_HIGH_B
# ]

ALL_ABM_SCENARIOS = [ 
    ABM_DENSE_LOW_B
]
# ALL_ABM_SCENARIOS = [TEST1, TEST2]

def get_abm_scenario(name: str) -> ABMScenarioConfig:
    """Get ABM scenario configuration by name"""
    for scenario in ALL_ABM_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Scenario '{name}' not found. Available: {[s.name for s in ALL_ABM_SCENARIOS]}")


# abm_config.py - Modified density sweep section

# ==================== Density Sweep Scenarios with Seed Ratio Dimension ====================
# Varying both network density AND initial seed ratio
# Purpose: Analyze interaction between connectivity and initial activation

def create_density_seed_sweep_scenarios(base_name_prefix: str,
                                        role_distribution: dict,
                                        densities: list,
                                        seed_ratios: list,
                                        base_seed: int = 3000) -> list:
    """
    Create scenarios varying both network density and initial seed ratio
    
    Args:
        base_name_prefix: Name prefix, e.g., "Sweep_HighResistant"
        role_distribution: Fixed role distribution dictionary
        densities: List of target average degrees, e.g., [2.0, 3.0, 4.0, 5.0]
        seed_ratios: List of initial seed ratios, e.g., [0.05, 0.10, 0.15, 0.20]
        base_seed: Starting random seed
    
    Returns:
        List of ABMScenarioConfig objects
    """
    scenarios = []
    scenario_idx = 0
    
    for density in densities:
        for seed_ratio in seed_ratios:
            scenarios.append(
                ABMScenarioConfig(
                    name=f"{base_name_prefix}_D{density:.1f}_S{int(seed_ratio*100)}",
                    description=f"{base_name_prefix}: Degree={density:.1f}, Seed={seed_ratio:.0%}",
                    target_avg_degree=density,
                    target_seed_ratio=seed_ratio,
                    target_bonding_ratio=0.25,
                    **role_distribution,
                    rho_structural_bonding=0.0,
                    rho_structural_bridging=0.0,
                    rho_temporal_bonding=0.5,
                    rho_temporal_bridging=0.5,
                    random_seed=base_seed + scenario_idx * 100,
                    n_timesteps=24
                )
            )
            scenario_idx += 1
    
    return scenarios

# Define 3 role distribution configurations (same as before)
ROLE_CONFIG_HIGH_RESISTANT = {
    'early_adopter_ratio': 0.10,
    'social_follower_ratio': 0.15,
    'resistant_ratio': 0.65,
    'isolated_ratio': 0.10
}

ROLE_CONFIG_BALANCED = {
    'early_adopter_ratio': 0.15,
    'social_follower_ratio': 0.25,
    'resistant_ratio': 0.45,
    'isolated_ratio': 0.15
}

ROLE_CONFIG_HIGH_SOCIAL = {
    'early_adopter_ratio': 0.20,
    'social_follower_ratio': 0.50,
    'resistant_ratio': 0.20,
    'isolated_ratio': 0.10
}

# Define sweep ranges
DENSITIES = [2.0, 3.0, 4.0, 5.0, 6.0]        # 5 density levels
SEED_RATIOS = [0.05, 0.20]       # 2 seed ratio levels

# Generate scenarios: 5 densities × 4 seed ratios = 20 per role config
DENSITY_SEED_SWEEP_HIGH_RESISTANT = create_density_seed_sweep_scenarios(
    "Sweep_HighResistant",
    ROLE_CONFIG_HIGH_RESISTANT,
    DENSITIES,
    SEED_RATIOS,
    base_seed=3000
)

DENSITY_SEED_SWEEP_BALANCED = create_density_seed_sweep_scenarios(
    "Sweep_Balanced",
    ROLE_CONFIG_BALANCED,
    DENSITIES,
    SEED_RATIOS,
    base_seed=4000
)

DENSITY_SEED_SWEEP_HIGH_SOCIAL = create_density_seed_sweep_scenarios(
    "Sweep_HighSocial",
    ROLE_CONFIG_HIGH_SOCIAL,
    DENSITIES,
    SEED_RATIOS,
    base_seed=5000
)

# Total: 3 role configs × 10 scenarios = 30 scenarios
ALL_DENSITY_SEED_SWEEP_SCENARIOS = (
    DENSITY_SEED_SWEEP_HIGH_RESISTANT + 
    DENSITY_SEED_SWEEP_BALANCED + 
    DENSITY_SEED_SWEEP_HIGH_SOCIAL
)