import pandas as pd
import numpy as np
import pdb

def generate_T0_states(house_df_with_features, T):
    # Initialize state dataframe: one row per household per time step
    # T = 24
    states = []

    # Compute community-level average damage
    community_damage = house_df_with_features.groupby('community')['damage_level'].mean().to_dict()

    for idx, row in house_df_with_features.iterrows():
        home_id = row['home']
        community = row['community']
        damage = row['damage_level']

        # At t = 0:
        # 1. Assign repair state: only possible if damage > 0
        if damage > 0:
            p_repair = min(0.1 + 0.1 * damage, 1.0)  # Higher damage → higher chance
            repair_0 = int(np.random.rand() < p_repair)
        else:
            repair_0 = int(np.random.rand() < 0.05)

        # 2. Assign vacancy state based on community damage level
        p_vacant = min(0.08 + 0.05 * community_damage.get(community, 0), 0.2)
        vacant_0 = int(np.random.rand() < p_vacant)

        # 3. Assign sales state similarly
        p_sales = min(0.02 + 0.05 * community_damage.get(community, 0), 0.2)
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


def update_full_states_one_step(house_df_with_features:pd.DataFrame,
                                full_states_df: pd.DataFrame,
                                p_self_series: pd.Series,
                                p_ji_df: pd.DataFrame,
                                links_df: pd.DataFrame,
                                t: int,
                                k: int) -> pd.DataFrame:
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
    full_states_df : DataFrame
        Same object, but states at time t+1 for dimension k are updated.
    """
    state_cols = ['vacancy_state', 'repair_state', 'sales_state']
    k_col      = state_cols[k]

    # --- 0. Slice current & next rows ---------------------------------------
    cur_df  = full_states_df[full_states_df['time'] == t].set_index('home')
    # next_df = full_states_df[full_states_df['time'] == t + 1].set_index('home').copy()
    next_df = cur_df.copy()
    homes   = cur_df.index.tolist()
    link_m  = links_df.values
    state_k = cur_df[k_col].values        # s_i^k(t)

    rng = np.random.default_rng()

    # --- 1. Update each household -------------------------------------------
    for i, h_i in enumerate(homes):
        if k == 1:
            damage_level = house_df_with_features.set_index('home').loc[h_i, 'damage_level']

            if damage_level == 0 and int(np.random.rand() > 0.05):
                # next_df.at[h_i, k_col] = 0
                continue
        other_state_cols = [col for j, col in enumerate(state_cols) if j != k]
        already_active = cur_df.loc[h_i, other_state_cols].sum() > 0
        if already_active:
            continue

        # (a) irreversible: once active -> remain 1
        if state_k[i] == 1:
            next_df.at[h_i, k_col] = 1
            continue

        # (b) assemble active neighbours j with s_j^k(t)=1 & link>0
        neighbours_idx = np.where((link_m[:, i] > 0) & (state_k == 1))[0]

        if neighbours_idx.size == 0:
            activate_prob = p_self_series.loc[h_i]
        else:
            prod_term = np.prod(1 - p_ji_df.iloc[neighbours_idx, i].values)
            activate_prob = 1 - (1 - p_self_series.loc[h_i]) * prod_term

        # (c) Bernoulli sampling
        next_df.at[h_i, k_col] = int(rng.random() < activate_prob)

    # --- 2. Write back to full_states_df ------------------------------------
    full_states_df.set_index(['home', 'time'], inplace=True)
    next_df.index = pd.MultiIndex.from_arrays([next_df.index, [t + 1] * len(next_df)], names=['home', 'time'])

    full_states_df.loc[next_df.index, k_col] = next_df[k_col].values

    full_states_df.reset_index(inplace=True)
    return full_states_df
