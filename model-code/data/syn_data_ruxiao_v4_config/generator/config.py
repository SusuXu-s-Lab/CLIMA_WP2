# config.py
from dataclasses import dataclass
import json
from typing import Dict, Any

@dataclass
class ScenarioConfig:
    """Configuration for a single scenario"""
    # Basic scenario information
    name: str
    description: str
    
    # Network density parameters (affect link generation)
    alpha_bonding: float = 1.3      # Your original alpha parameter in main.py
    beta_bridging: float = 0.0001   # Your original beta parameter in main.py
    gamma_decay: float = 0.6        # Your original gamma parameter in main.py
    target_seed_ratio: float = 0.07    # default: 7% households seeded at t=0
    jitter_fraction: float = 0.2       # fraction for randomization around cutoff
    
    # Decision seeding parameters (affect initial decision ratios)
    repair_base_prob: float = 0.05
    repair_damage_coeff: float = 0.35
    repair_building_coeff: float = 0.2
    repair_income_coeff: float = 0.1
    
    vacant_base_prob: float = 0.1
    vacant_damage_coeff: float = 0.3
    vacant_income_coeff: float = 0.2
    vacant_age_coeff: float = 0.1
    
    sales_base_prob: float = 0.1 * 0.5
    sales_damage_coeff: float = 0.25 * 0.5
    sales_building_coeff: float = 0.3 * 0.5
    sales_age_coeff: float = 0.1  * 0.5
    
    # Simulation parameters
    n_households: int = 200
    time_horizon: int = 24
    p_block: float = 0.5
    blocking_mode: str = "temporal"
    L: int = 1
    target_avg_degree: float = 5.0  # Desired average degree for the network
    random_seed: int = 42

    # --- Soft inverse-U time gate (not a hard cutoff) ---
    decision_start: dict = None   # e.g. {'repair':0,'vacancy':2,'sales':4}
    decision_peak:  dict = None   # e.g. {'repair':4,'vacancy':8,'sales':10}
    time_gate_floor: float = 0.05 # allow small early/late activations
    time_decay_rate: float = 0.10 # post-peak exponential decay

    # --- Cross-decision dependency (non-exclusive, multiplicative) ---
    # If (d1 at time t == 1), multiply current prob of d2 by this factor
    decision_dependency: dict = None

    # --- Geographic neighbor restriction for link formation ---
    top_k_neighbors: int = 100    # only try edges within K nearest neighbors
    geo_hard_cutoff: float = 0.0  # optional hard distance cutoff (meters); 0 means disabled

def fill_missing_defaults(cfg):
    """Supply gentle defaults so older configs continue to work."""
    if cfg.decision_start is None:
        cfg.decision_start = {'repair': 0, 'vacancy': 2, 'sales': 4}
    if cfg.decision_peak is None:
        cfg.decision_peak  = {'repair': 4, 'vacancy': 8, 'sales': 10}
    if cfg.decision_dependency is None:
        cfg.decision_dependency = {
            ('repair','vacancy'): 0.8,
            ('repair','sales'):   0.5,
            ('vacancy','sales'):  1.3,
            ('sales','vacancy'):  1.5,
        }
    if cfg.top_k_neighbors is None:
        cfg.top_k_neighbors = 100
    if cfg.geo_hard_cutoff is None:
        cfg.geo_hard_cutoff = 0.0
    if cfg.time_gate_floor is None:
        cfg.time_gate_floor = 0.05
    if cfg.time_decay_rate is None:
        cfg.time_decay_rate = 0.10
    return cfg

# Define 6 predefined scenarios
G1_SPARSE_LOW = ScenarioConfig(
    name="G1_Sparse_LowSeed",
    description="Sparse network with low initial decision seeding (8-15%)",
    alpha_bonding=0.5,      # Lower alpha = sparser network
    beta_bridging=1.1,  # Lower beta = sparser network
    repair_base_prob=0.00005, # Lower base probs = low seeding
    vacant_base_prob=0.00005,
    sales_base_prob=0.00005,
    repair_damage_coeff=0.15,  # Lower coeffs = less influence
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio = 0.05,   
    jitter_fraction = 0.2,   
    target_avg_degree = 5.0,
    blocking_mode="temporal",    
    random_seed=1000
)

G2_SPARSE_HIGH = ScenarioConfig(
    name="G2_Sparse_HighSeed", 
    description="Sparse network with high initial decision seeding (20-25%)",
    alpha_bonding=0.5,      # Lower alpha = sparser network
    beta_bridging=1.1,  # Lower beta = sparser network
    repair_base_prob=0.08,  # Higher base probs = high seeding
    vacant_base_prob=0.06,
    sales_base_prob=0.1,
    repair_damage_coeff=0.4,  # Higher coeffs = more influence
    vacant_damage_coeff=0.3,
    sales_damage_coeff=0.35,
    target_seed_ratio = 0.15,   
    jitter_fraction = 0.2, 
    target_avg_degree = 5.0, 
    blocking_mode="temporal",    
    random_seed=1000
)

G3_MEDIUM_LOW = ScenarioConfig(
    name="G3_Medium_LowSeed",
    description="Medium density network with low initial decision seeding (8-15%)",
    alpha_bonding=0.6,      # Medium alpha = medium density
    beta_bridging=1.5,   # Medium beta = medium density
    repair_base_prob=0.005, # Low seeding
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.15,
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio = 0.05,   
    jitter_fraction = 0.2,    
    target_avg_degree = 8.0,
    blocking_mode="temporal",  
    random_seed=1003
)

G4_MEDIUM_HIGH = ScenarioConfig(
    name="G4_Medium_HighSeed",
    description="Medium density network with high initial decision seeding (20-25%)",
    alpha_bonding=0.6,      # Medium alpha = medium density
    beta_bridging=1.5,   # Medium beta = medium density
    repair_base_prob=0.08,  # High seeding
    vacant_base_prob=0.06,
    sales_base_prob=0.1,
    repair_damage_coeff=0.4,
    vacant_damage_coeff=0.3,
    sales_damage_coeff=0.35,
    target_seed_ratio = 0.15,   
    jitter_fraction = 0.2,    
    target_avg_degree = 8.0,
    blocking_mode="temporal",  
    random_seed=1004
)

G5_DENSE_LOW = ScenarioConfig(
    name="G5_Dense_LowSeed",
    description="Dense network with low initial decision seeding (8-15%)",
    alpha_bonding=0.7,      # Higher alpha = denser network
    beta_bridging=1.8,   # Higher beta = denser network
    repair_base_prob=0.005, # Low seeding
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.15,
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio = 0.05,   
    jitter_fraction = 0.2,   
    target_avg_degree = 12.0,
    blocking_mode="temporal",   
    random_seed=1005
)

G6_DENSE_HIGH = ScenarioConfig(
    name="G6_Dense_HighSeed",
    description="Dense network with high initial decision seeding (15-25%)",
    alpha_bonding=0.7,      # Higher alpha = denser network
    beta_bridging=1.8,   # Higher beta = denser network
    repair_base_prob=0.08,  # High seeding
    vacant_base_prob=0.06,
    sales_base_prob=0.1,
    repair_damage_coeff=0.4,
    vacant_damage_coeff=0.3,
    sales_damage_coeff=0.35,
    target_seed_ratio = 0.15,   
    jitter_fraction = 0.2,   
    target_avg_degree = 12.0, 
    blocking_mode="temporal",  
    random_seed=1006
)

# Structural blocking versions of all scenarios
G1_SPARSE_LOW_STRUCT = ScenarioConfig(
    name="G1_Sparse_LowSeed_Structural",
    description="Sparse network with low initial decision seeding (8-15%) - Structural blocking",
    alpha_bonding=0.5,
    beta_bridging=1.1,
    repair_base_prob=0.00005,
    vacant_base_prob=0.00005,
    sales_base_prob=0.00005,
    repair_damage_coeff=0.15,
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio=0.15,
    jitter_fraction=0.2,
    target_avg_degree = 5.0,
    blocking_mode="structural",  # Key difference
    random_seed=1089
)

G2_SPARSE_HIGH_STRUCT = ScenarioConfig(
    name="G2_Sparse_HighSeed_Structural",
    description="Sparse network with high initial decision seeding (20-25%) - Structural blocking",
    alpha_bonding=0.5,
    beta_bridging=1.1,
    repair_base_prob=0.08,
    vacant_base_prob=0.06,
    sales_base_prob=0.1,
    repair_damage_coeff=0.4,
    vacant_damage_coeff=0.3,
    sales_damage_coeff=0.35,
    target_seed_ratio=0.25,
    jitter_fraction=0.2,
    target_avg_degree = 5.0,
    blocking_mode="structural",
    random_seed=1002
)

G3_MEDIUM_LOW_STRUCT = ScenarioConfig(
    name="G3_Medium_LowSeed_Structural",
    description="Medium density network with low initial decision seeding (8-15%) - Structural blocking",
    alpha_bonding=0.6,
    beta_bridging=1.5,
    repair_base_prob=0.005,
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.15,
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio=0.15,
    jitter_fraction=0.2,
    target_avg_degree = 8.0,
    blocking_mode="structural",
    random_seed=1003
)

G4_MEDIUM_HIGH_STRUCT = ScenarioConfig(
    name="G4_Medium_HighSeed_Structural",
    description="Medium density network with high initial decision seeding (20-25%) - Structural blocking",
    alpha_bonding=0.6,
    beta_bridging=1.5,
    repair_base_prob=0.08,
    vacant_base_prob=0.06,
    sales_base_prob=0.1,
    repair_damage_coeff=0.4,
    vacant_damage_coeff=0.3,
    sales_damage_coeff=0.35,
    target_seed_ratio=0.25,
    jitter_fraction=0.2,
    target_avg_degree = 8.0,
    blocking_mode="structural",
    random_seed=1004
)

G5_DENSE_LOW_STRUCT = ScenarioConfig(
    name="G5_Dense_LowSeed_Structural",
    description="Dense network with low initial decision seeding (8-15%) - Structural blocking",
    alpha_bonding=0.7,
    beta_bridging=1.8,
    repair_base_prob=0.005,
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.15,
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio=0.15,
    jitter_fraction=0.2,
    target_avg_degree = 10.0,
    blocking_mode="structural",
    random_seed=1005
)

G6_DENSE_HIGH_STRUCT = ScenarioConfig(
    name="G6_Dense_HighSeed_Structural",
    description="Dense network with high initial decision seeding (15-25%) - Structural blocking",
    alpha_bonding=0.7,
    beta_bridging=1.8,
    repair_base_prob=0.08,
    vacant_base_prob=0.06,
    sales_base_prob=0.1,
    repair_damage_coeff=0.4,
    vacant_damage_coeff=0.3,
    sales_damage_coeff=0.35,
    target_seed_ratio=0.25,
    jitter_fraction=0.2,
    target_avg_degree = 10.0,
    blocking_mode="structural",
    random_seed=1006
)

G7_SUPER_DENSE_LOW = ScenarioConfig(
    name="G7_Super_Dense_LowSeed",
    description="Super Dense network with low initial decision seeding (8-15%)",
    alpha_bonding=0.7,
    beta_bridging=1.8,
    repair_base_prob=0.005,
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.13,
    vacant_damage_coeff=0.55,
    sales_damage_coeff=0.4,
    target_seed_ratio=0.10,
    jitter_fraction=0.2,
    target_avg_degree = 20.0,
    blocking_mode="temporal",
    random_seed=1046
)

G8_SUPER_DENSE_HIGH = ScenarioConfig(
    name="G8_Super_Dense_HighSeed",
    description="Super Dense network with high initial decision seeding (15-25%)",
    alpha_bonding=0.7,
    beta_bridging=1.8,
    repair_base_prob=0.005,
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.13,
    vacant_damage_coeff=0.45,
    sales_damage_coeff=0.4,
    target_seed_ratio=0.20,
    jitter_fraction=0.2,
    target_avg_degree = 20.0,
    blocking_mode="temporal",
    random_seed=1016
)

G1_SPARSE_LOW_n50 = ScenarioConfig(
    name="G1_Sparse_LowSeed_n50",
    description="Sparse network with low initial decision seeding (8-15%)",
    n_households=50,
    alpha_bonding=0.5,      # Lower alpha = sparser network
    beta_bridging=1.1,  # Lower beta = sparser network
    repair_base_prob=0.00005, # Lower base probs = low seeding
    vacant_base_prob=0.00005,
    sales_base_prob=0.00005,
    repair_damage_coeff=0.15,  # Lower coeffs = less influence
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio = 0.05,   
    jitter_fraction = 0.2,   
    target_avg_degree = 5.0,
    blocking_mode="temporal",    
    random_seed=1000
)

G5_DENSE_LOW_n50 = ScenarioConfig(
    name="G5_Dense_LowSeed_n50",
    description="Dense network with low initial decision seeding (8-15%) ",
    n_households=50,
    alpha_bonding=0.7,
    beta_bridging=1.8,
    repair_base_prob=0.005,
    vacant_base_prob=0.005,
    sales_base_prob=0.005,
    repair_damage_coeff=0.15,
    vacant_damage_coeff=0.1,
    sales_damage_coeff=0.1,
    target_seed_ratio=0.15,
    jitter_fraction=0.2,
    target_avg_degree = 10.0,
    blocking_mode="structural",
    random_seed=1005
)

ALL_SCENARIOS = [
#     # Temporal blocking scenarios (original 6)
    G1_SPARSE_LOW, G2_SPARSE_HIGH,
    G3_MEDIUM_LOW, G4_MEDIUM_HIGH,
    G5_DENSE_LOW, G6_DENSE_HIGH,
    
#     # Structural blocking scenarios (new 6)
#     G1_SPARSE_LOW_STRUCT, G2_SPARSE_HIGH_STRUCT,
#     G3_MEDIUM_LOW_STRUCT, G4_MEDIUM_HIGH_STRUCT,
#     G5_DENSE_LOW_STRUCT, G6_DENSE_HIGH_STRUCT
]
ALL_SCENARIOS = [G1_SPARSE_LOW_n50, G5_DENSE_LOW_n50]


# ALL_SCENARIOS = [G1_SPARSE_LOW, G2_SPARSE_HIGH,G1_SPARSE_LOW_STRUCT, G2_SPARSE_HIGH_STRUCT]
# ALL_SCENARIOS = [G8_SUPER_DENSE_HIGH]

def get_scenario(name: str) -> ScenarioConfig:
    """Get scenario configuration by name"""
    for scenario in ALL_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Scenario '{name}' not found. Available: {[s.name for s in ALL_SCENARIOS]}")

# config.py - Add at the end

# ==================== Generator Density×Seed Sweep ====================
def create_generator_density_seed_sweep(base_name_prefix: str,
                                       base_params: dict,
                                       densities: list,
                                       seed_ratios: list,
                                       base_seed: int = 9000) -> list:
    """
    Create generator scenarios varying density and seed ratio
    
    Args:
        base_name_prefix: Name prefix
        base_params: Base parameters (seeding coefficients, etc.)
        densities: List of target average degrees
        seed_ratios: List of target seed ratios
        base_seed: Starting random seed
    
    Returns:
        List of ScenarioConfig objects
    """
    scenarios = []
    scenario_idx = 0
    
    for density in densities:
        for seed_ratio in seed_ratios:
            scenarios.append(
                ScenarioConfig(
                    name=f"{base_name_prefix}_D{density:.1f}_S{int(seed_ratio*100)}",
                    description=f"Generator: Degree={density:.1f}, Seed={seed_ratio:.0%}",
                    target_avg_degree=density,
                    target_seed_ratio=seed_ratio,
                    **base_params,
                    blocking_mode="temporal",
                    random_seed=base_seed + scenario_idx * 100
                )
            )
            scenario_idx += 1
    
    return scenarios

# Base parameters for generator (similar across all configs)
GEN_BASE_PARAMS = {
    'alpha_bonding': 0.6,
    'beta_bridging': 1.5,
    'gamma_decay': 0.6,
    'repair_base_prob': 0.005,
    'repair_damage_coeff': 0.25,
    'repair_building_coeff': 0.2,
    'repair_income_coeff': 0.1,
    'vacant_base_prob': 0.02,
    'vacant_damage_coeff': 0.2,
    'vacant_income_coeff': 0.2,
    'vacant_age_coeff': 0.1,
    'sales_base_prob': 0.01,
    'sales_damage_coeff': 0.2,
    'sales_building_coeff': 0.2,
    'sales_age_coeff': 0.1,
    'jitter_fraction': 0.2,
    'n_households': 200,
    'time_horizon': 24,
    'p_block': 0.5,
    'L': 1
}

# Sweep ranges (same as ABM)
GEN_DENSITIES = [2.0, 3.0, 4.0, 5.0, 6.0]
GEN_SEED_RATIOS = [0.05, 0.20]

# Generate 10 scenarios (5 densities × 2 seed ratios)
ALL_GEN_DENSITY_SEED_SWEEP = create_generator_density_seed_sweep(
    "GenSweep",
    GEN_BASE_PARAMS,
    GEN_DENSITIES,
    GEN_SEED_RATIOS,
    base_seed=9000
)