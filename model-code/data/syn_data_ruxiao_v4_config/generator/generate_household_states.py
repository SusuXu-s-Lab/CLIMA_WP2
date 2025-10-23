import pandas as pd
import numpy as np
import pdb
from config import ScenarioConfig

def time_gate(decision_name: str, t: int, T: int,
              start_map: dict, peak_map: dict,
              floor: float = 0.05, decay: float = 0.10) -> float:
    """
    Soft inverse-U time gate (not a hard cutoff).
    - Before 'start': return a small floor so early activations are possible but rare.
    - From start to peak: smoothly ramp up to ~1 using a half-sine curve.
    - After peak: exponential decay; never drop below 'floor'.
    """
    start = start_map.get(decision_name, 0)
    peak  = peak_map.get(decision_name, max(1, T // 2))

    if t <= start:
        return floor

    if t <= peak:
        # Smooth ramp-up with a half-sine: sin(π/2 * progress), progress ∈ [0,1]
        progress = (t - start) / max(1.0, (peak - start))
        return floor + (1.0 - floor) * np.sin(0.5 * np.pi * np.clip(progress, 0.0, 1.0))

    # Post-peak exponential decay
    return max(floor, np.exp(-decay * (t - peak)))


def generate_T0_states_with_config(house_df_with_features: pd.DataFrame, T: int, config: ScenarioConfig):
    """
    Generate initial household states using configuration parameters
    
    Args:
        house_df_with_features: DataFrame with household features
        T: Time horizon
        config: ScenarioConfig object with probability parameters
        
    Returns:
        DataFrame with initial states for all households across all time steps
    """
    # Initialize state dataframe: one row per household per time step
    states = []

    for idx, row in house_df_with_features.iterrows():
        home_id = row['home']
        
        # Extract household features for linear combination
        building_value = row['building_value']
        income = row['income'] 
        damage_level = row['damage_level']
        population_scaled = row['population_scaled']
        age = row['age']
        race = row['race']

        # At t = 0: Assign repair state using config parameters
        p_repair = min(
            config.repair_base_prob + 
            config.repair_damage_coeff * damage_level + 
            config.repair_building_coeff * building_value + 
            config.repair_income_coeff * income, 
            1.0
        )
        repair_0 = int(np.random.rand() < p_repair)

        # Assign vacancy state using config parameters
        p_vacant = min(
            config.vacant_base_prob + 
            config.vacant_damage_coeff * damage_level + 
            config.vacant_income_coeff * income + 
            config.vacant_age_coeff * age, 
            1.0
        )
        vacant_0 = int(np.random.rand() < p_vacant)

        # Assign sales state using config parameters
        p_sales = min(
            config.sales_base_prob + 
            config.sales_damage_coeff * damage_level + 
            config.sales_building_coeff * building_value + 
            config.sales_age_coeff * (1 - age), 
            1.0
        )
        sales_0 = int(np.random.rand() < p_sales)

        for t in range(T):
            states.append({
                'home': home_id,
                'time': t,
                'repair_state': repair_0 if t == 0 else 0,
                'vacancy_state': vacant_0 if t == 0 else 0,
                'sales_state': sales_0 if t == 0 else 0
            })

    # Convert to DataFrame
    states_df = pd.DataFrame(states)
    return states_df



def generate_T0_states_with_config(house_df_with_features: pd.DataFrame, T: int, config: ScenarioConfig):
    """
    Generate initial household states at t=0 with mutual exclusivity and controlled ratio.
    Each household is either none/repair/vacant/sales, chosen by logistic probabilities
    + top-k with jitter. States at later timesteps are initialized to 0.
    
    Args:
        house_df_with_features: DataFrame with household features
        T: Time horizon
        config: ScenarioConfig object with probability parameters + target_seed_ratio
        
    Returns:
        DataFrame with initial states for all households across all time steps
    """
    import numpy as np
    import pandas as pd

    N = len(house_df_with_features)

    # Logistic helper
    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Compute raw probabilities for each decision type
    p_repair = logistic(
        config.repair_base_prob
        + config.repair_damage_coeff * house_df_with_features['damage_level'].values
        + config.repair_building_coeff * house_df_with_features['building_value'].values
        + config.repair_income_coeff * house_df_with_features['income'].values
    )
    p_vacant = logistic(
        config.vacant_base_prob
        + config.vacant_damage_coeff * house_df_with_features['damage_level'].values
        + config.vacant_income_coeff * house_df_with_features['income'].values
        + config.vacant_age_coeff * house_df_with_features['age'].values
    )
    p_sales = logistic(
        config.sales_base_prob
        + config.sales_damage_coeff * house_df_with_features['damage_level'].values
        + config.sales_building_coeff * house_df_with_features['building_value'].values
        + config.sales_age_coeff * (1.0 - house_df_with_features['age'].values)
    )

    # Stage 1: decide which households are seeds (global control)
    scores = p_repair + p_vacant + p_sales
    k = int(np.round(config.target_seed_ratio * N))
    m = int(config.jitter_fraction * k)

    idx_sorted = np.argsort(-scores)  # descending order
    chosen = idx_sorted[:max(k - m, 0)].tolist()
    candidate_pool = idx_sorted[max(k - m, 0):min(k + m, N)]
    if len(candidate_pool) > 0 and m > 0:
        chosen += list(np.random.choice(candidate_pool, size=min(m, len(candidate_pool)), replace=False))

    seed_mask = np.zeros(N, dtype=bool)
    seed_mask[chosen] = True

    # Stage 2: assign mutually exclusive type (repair/vacant/sales) for chosen households
    repair_state_0 = np.zeros(N, dtype=int)
    vacant_state_0 = np.zeros(N, dtype=int)
    sales_state_0 = np.zeros(N, dtype=int)

    for i in chosen:
        probs = np.array([p_repair[i], p_vacant[i], p_sales[i]])
        probs = probs / probs.sum()
        decision = np.random.choice(3, p=probs)
        if decision == 0:
            repair_state_0[i] = 1
        elif decision == 1:
            vacant_state_0[i] = 1
        else:
            sales_state_0[i] = 1

    # Build output DataFrame (states for all timesteps)
    states = []
    for idx, row in house_df_with_features.iterrows():
        home_id = row['home']
        for t in range(T):
            states.append({
                'home': home_id,
                'time': t,
                'repair_state': int(repair_state_0[idx]) if t == 0 else 0,
                'vacancy_state': int(vacant_state_0[idx]) if t == 0 else 0,
                'sales_state': int(sales_state_0[idx]) if t == 0 else 0
            })

    return pd.DataFrame(states)



def update_full_states_one_step(house_df_with_features:pd.DataFrame,
                                full_states_df: pd.DataFrame,
                                p_self_series: pd.Series,
                                p_ji_df: pd.DataFrame,
                                links_df: pd.DataFrame,
                                t: int,
                                k: int,
                                config=None) -> tuple[pd.DataFrame, list]:
    """
    Update full_states_df in-place for time t+1, dimension k.

    Parameters
    ----------
    full_states_df : DataFrame
        Columns ['home','time','repair_state','vacancy_state','sales_state'].
        Rows for all times 0..T already exist (future rows • value 0 placeholder).
    p_self_series : Series
        p_self_i^k(t) indexed by household 'home'.
    p_ji_df : DataFrame
        p_{ji}^k(t) matrix (rows = j, cols = i).  Must use same home order as links_df.
    links_df : DataFrame
        Symmetric link matrix ℓ_{ij}(t) (values 0/1/2) for current step t.
    t : int
        Current time step (states at time t must already be filled).
    k : int
        Target state dimension (0=repair, 1=vacancy, 2=sales).

    Returns
    -------
    tuple : (full_states_df, activation_records)
        full_states_df : DataFrame - Same object, but states at time t+1 for dimension k are updated.
        activation_records : list - List of dictionaries containing prod_term and activate_prob values.
    """
    state_cols = ['vacancy_state', 'repair_state', 'sales_state']
    k_col      = state_cols[k]

    # --- Initialize recording list for this step ----------------------------
    activation_records = []

    # --- 0. Slice current & next rows ---------------------------------------
    cur_df  = full_states_df[full_states_df['time'] == t].set_index('home')
    # next_df = full_states_df[full_states_df['time'] == t + 1].set_index('home').copy()
    next_df = cur_df.copy()
    homes   = cur_df.index.tolist()
    link_m  = links_df.values
    state_k = cur_df[k_col].values        # s_i^k(t)


    # --- 1. Update each household -------------------------------------------
    for i, h_i in enumerate(homes):
        if k == 1:
            damage_level = house_df_with_features.set_index('home').loc[h_i, 'damage_level']

            if damage_level == 0:
                # next_df.at[h_i, k_col] = 0
                continue
        # other_state_cols = [col for j, col in enumerate(state_cols) if j != k]
        # already_active = cur_df.loc[h_i, other_state_cols].sum() > 0
        # if already_active:
        #     continue

        # (a) irreversible: once active -> remain 1
        if state_k[i] == 1:
            next_df.at[h_i, k_col] = 1
            continue

        # (b) assemble active neighbours j with s_j^k(t)=1 & link>0
        neighbours_idx = np.where((link_m[:, i] > 0) & (state_k == 1))[0]

        if neighbours_idx.size == 0:
            prod_term = 1.0  # No neighbors, so product term is 1
            activate_prob = p_self_series.loc[h_i]
        else:
            prod_term = np.prod(1 - p_ji_df.iloc[neighbours_idx, i].values)
            activate_prob = 1 - (1 - p_self_series.loc[h_i]) * prod_term

        # === [ADD HERE] Soft time gate + dependency multipliers ===
        # Map column to decision name (for readability)
        name_map = {
            'repair_state':  'repair',
            'vacancy_state': 'vacancy',
            'sales_state':   'sales'
        }
        decision_name = name_map[k_col]

        # (1) Soft time gate (inverse-U shaped multiplier)
        T_total = full_states_df['time'].max() + 1
        tg = time_gate(
            decision_name,
            t,
            T_total,
            start_map=config.decision_start,
            peak_map=config.decision_peak,
            floor=config.time_gate_floor,
            decay=config.time_decay_rate
        )
        activate_prob *= tg  # scale activation by time window

        # (2) Cross-decision dependency (multiplicative, non-exclusive)
        dep_factor = 1.0
        other_cols = [c for c in state_cols if c != k_col]
        other_names = [name_map[c] for c in other_cols]
        other_vals = cur_df.loc[h_i, other_cols].values  # values of other decisions at t

        for val, oname in zip(other_vals, other_names):
            if val == 1:
                dep_factor *= config.decision_dependency.get((oname, decision_name), 1.0)

        activate_prob *= dep_factor
        # === [END ADDITION] ===
        
        activate_prob = np.minimum(activate_prob, 0.8)  # Cap at 0.5


        # Record the values for analysis
        activation_records.append({
            'time_step': t,
            'dimension': k,
            'dimension_name': state_cols[k],
            'household_id': h_i,
            'num_active_neighbors': len(neighbours_idx),
            'prod_term': prod_term,
            'activate_prob': activate_prob,
            'p_self': p_self_series.loc[h_i]
        })

        # (c) Probabilistic sampling: activate based on probability
        next_df.at[h_i, k_col] = int(np.random.rand() < activate_prob)
        # print(f"h_i: {h_i}, activate_prob: {activate_prob}, next_df.at[h_i, k_col]: {next_df.at[h_i, k_col]}")

    # --- 2. Write back to full_states_df ------------------------------------
    full_states_df.set_index(['home', 'time'], inplace=True)
    next_df.index = pd.MultiIndex.from_arrays([next_df.index, [t + 1] * len(next_df)], names=['home', 'time'])

    full_states_df.loc[next_df.index, k_col] = next_df[k_col].values

    full_states_df.reset_index(inplace=True)
    return full_states_df, activation_records
