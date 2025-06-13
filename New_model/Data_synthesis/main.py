import pdb
from generate_household_features import generate_household_features
from generate_household_states import generate_T0_states
import pandas as pd
from household_features_function import compute_similarity, compute_interaction_potential
from links_updates_fun import generate_initial_links
from tqdm import tqdm
import math
import json, itertools


# Hyper parameter definition
alpha=0.9
beta=0.5
L=1

'''
Read real Household Features
'''
# Read T=0 partially observed social links
df_ori = pd.read_csv('household_swinMaryland_20190101.csv')
df_ori = df_ori[['home_1', 'home_2', 'home_1_number', 'home_2_number']]

# Extract home and their locations
home1 = df_ori[['home_1']].rename(columns={'home_1': 'home'})
home2 = df_ori[['home_2']].rename(columns={'home_2': 'home'})

# Merge and remove duplicates
house_df = pd.concat([home1, home2], ignore_index=True).drop_duplicates(subset='home')
house_df = house_df.dropna()

'''
Household Feautres Generation
'''
house_df_with_features = generate_household_features(house_df)

'''
T=0 Household States Generation
'''
house_states=generate_T0_states(house_df_with_features,24)

'''
T=0 Household Similarity and Intercation Potential
'''
# Compute Similarity and Interaction Potential
similarity_df = compute_similarity(house_df_with_features)
interaction_df = compute_interaction_potential(house_df_with_features, house_states, t=0)

'''
T=0 Links Generation
'''
# Generate initial link matrix using t=0 similarity and interaction matrices
initial_links_df = generate_initial_links(similarity_df, interaction_df, alpha_0=alpha, beta_0=beta)

import numpy as np

def compute_p_self(house_df, full_state_df, t, k, L=3):
    """
    Compute p_self^k_i(t) for each household i using linear features + sigmoid.

    Args:
        house_df: DataFrame with static features, indexed by 'home'
        full_state_df: DataFrame of all states across time
        t: current time step
        k: state dimension (0=repair, 1=vacancy, 2=sales)
        L: number of past steps to consider

    Returns:
        Series of p_self^k_i(t), indexed by household ID
    """
    x_i = house_df[['income', 'age', 'race']].copy()
    state_cols = ['repair_state', 'vacancy_state', 'sales_state']
    non_k_cols = [col for i, col in enumerate(state_cols) if i != k]

    # Collect state history from t-L to t-1
    frames = []
    for delta in range(1, L + 1):
        time_lookup = t - delta
        if time_lookup >= 0:
            df_slice = full_state_df[full_state_df['time'] == time_lookup].copy()
        else:
            df_slice = pd.DataFrame(columns=['home', 'time'] + state_cols)
        frames.append(df_slice)

    state_hist_df = pd.concat(frames, ignore_index=True)
    s_mean = state_hist_df.groupby('home')[non_k_cols].mean()

    # Combine features
    x_all = pd.concat([x_i, s_mean], axis=1)
    x_all['time'] = t
    x_all = x_all.fillna(0)

    weights = np.array([1.0, 0.5, 0.5, -0.8, -0.8, 0.01])  # shape (6,)
    dot = x_all.values @ weights
    p_self = 1 / (1 + np.exp(-dot))

    return pd.Series(p_self, index=x_all.index)


def compute_p_ji(link_df, house_df, state_df, similarity_df, interaction_df, k):
    """
    Compute p_ji^k(t) for all (i, j) pairs with links, using linear feature + sigmoid.

    Args:
        link_df: DataFrame of current links (values in {0,1,2})
        house_df: household static features
        state_df: current state (repair/vacancy/sales)
        similarity_df: precomputed similarity matrix
        interaction_df: precomputed interaction matrix
        k: target state dimension

    Returns:
        Dictionary of {(j, i): p_ji^k(t)} for all linked (j, i)
    """
    homes = house_df['home'].tolist()
    home_idx = {h: i for i, h in enumerate(homes)}
    s_k = state_df.set_index('home')[['repair_state', 'vacancy_state', 'sales_state']].values[:, k]
    p_ji_dict = {}

    for i in range(len(homes)):
        for j in range(len(homes)):
            if i == j:
                continue
            link_type = link_df.iloc[i, j]
            if link_type == 0:
                continue

            h_i, h_j = homes[i], homes[j]
            demo_diff = np.abs(
                house_df.loc[h_i, ['income', 'age', 'race']] - house_df.loc[h_j, ['income', 'age', 'race']])
            f_ij = demo_diff.values
            s_jk = s_k[j]
            s_ik = s_k[i]
            dist_ij = 1 - similarity_df.iloc[i, j]  # approximate inverse similarity
            inter_ij = interaction_df.iloc[i, j]

            # Feature vector: [demo_diff, s_jk, s_ik, link_type, dist_ij, interaction]
            features = np.concatenate([f_ij, [s_jk, s_ik, link_type, dist_ij, inter_ij]])
            weights = np.array([-2, -1, -1, 1.2, -1.2, 0.5, -0.5, 1.0])  # hand-defined
            score = np.dot(features, weights)
            p = 1 / (1 + np.exp(-score))
            p_ji_dict[(j, i)] = p

    return p_ji_dict

### Compute p_self^k_i(t)
# k: state dimension (0=repair, 1=vacancy, 2=sales)
t = 0
k = 0

p_self_series = compute_p_self(house_df_with_features.set_index('home'), house_states, t, k, L=1)
pdb.set_trace()
