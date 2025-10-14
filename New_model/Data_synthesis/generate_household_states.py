import pandas as pd
import numpy as np
import pdb

def generate_T0_states(house_df_with_features, T):
    # Initialize state dataframe: one row per household per time step
    # T = 24
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

        # At t = 0:
        # 1. Assign repair state: based on damage, building value, and income
        # Higher damage, higher building value, and higher income → higher repair probability
        p_repair = min(0.05 + 0.35 * damage_level + 0.2 * building_value + 0.1 * income, 1.0)
        repair_0 = int(p_repair > 0.5)

        # 2. Assign vacancy state: based on damage, low income, and age
        # Higher damage, lower income, older age → higher vacancy probability
        p_vacant = min(0.07 + 0.2 * damage_level + 0.2 * (1 - income) + 0.1 * age, 1.0)
        vacant_0 = int(p_vacant > 0.5)

        # 3. Assign sales state: based on damage, building value, and age
        # Higher damage, higher building value, younger age → higher sales probability
        p_sales = min(0.1 + 0.25 * damage_level + 0.3 * building_value + 0.1 * (1 - age), 1.0)
        sales_0 = int(p_sales > 0.5)

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


def update_full_states_one_step(house_df_with_features:pd.DataFrame,
                                full_states_df: pd.DataFrame,
                                p_self_series: pd.Series,
                                p_ji_df: pd.DataFrame,
                                links_df: pd.DataFrame,
                                t: int,
                                k: int, influence_counter: pd.DataFrame, x_times: int) -> tuple[pd.DataFrame, list]:
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
        
        # each household can only be active in one state dimension
        # other_state_cols = [col for j, col in enumerate(state_cols) if j != k]
        # already_active = cur_df.loc[h_i, other_state_cols].sum() > 0
        # if already_active:
        #     continue

        # (a) irreversible: once active -> remain 1
        if state_k[i] == 1:
            next_df.at[h_i, k_col] = 1
            continue

        # (b) assemble active neighbours j with s_j^k(t)=1 & link>0
                # (b) Only consider neighbors that are active AND have not exceeded influence limit
        valid_influencers_mask = (
            (state_k == 1) &
            (influence_counter.loc[homes, f'count_dim{k}'].values < x_times)
        )
        neighbours_idx = np.where((link_m[:, i] > 0) & valid_influencers_mask)[0]

        if neighbours_idx.size == 0:
            prod_term = 1.0  # No neighbors, so product term is 1
            activate_prob = p_self_series.loc[h_i]
        else:
            prod_term = np.prod(1 - p_ji_df.iloc[neighbours_idx, i].values)
            activate_prob = 1 - (1 - p_self_series.loc[h_i]) * prod_term
        
        activate_prob = np.minimum(activate_prob, 0.8)  # Cap at 0.8


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

    # --- 2. Write back to full_states_df ------------------------------------
    full_states_df.set_index(['home', 'time'], inplace=True)
    next_df.index = pd.MultiIndex.from_arrays([next_df.index, [t + 1] * len(next_df)], names=['home', 'time'])

    full_states_df.loc[next_df.index, k_col] = next_df[k_col].values

    full_states_df.reset_index(inplace=True)

    # --- 3. Update influence counters ---------------------------------------
    active_nodes = [homes[j] for j in np.where(state_k == 1)[0]]
    influence_counter.loc[active_nodes, f'count_dim{k}'] += 1
    influence_counter[f'count_dim{k}'] = influence_counter[f'count_dim{k}'].clip(upper=x_times)
    return full_states_df, activation_records
