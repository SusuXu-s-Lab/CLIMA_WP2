import pdb
import pygeohash as pgh
from generate_household_features import generate_household_features
from generate_household_states import generate_T0_states, update_full_states_one_step
import pandas as pd
from household_features_function import compute_similarity, compute_interaction_potential
from links_updates_fun import generate_initial_links, compute_p_self, compute_p_ji_linear, update_link_matrix_one_step
import numpy as np
from tqdm import tqdm
import os
import warnings
from probability_logger import log_generator_probabilities
warnings.filterwarnings("ignore")

# Hyper parameter definition
alpha=0.1
beta=0.0001
gamma=0.6
L=1
state_dims = ['vacancy_state', 'repair_state', 'sales_state']
T = 24          # total horizon (t = 0 … T-1  →  produce T steps of NEW state)
p_block=0.5 # propotion of links are randomly blocked per time step


'''
Read real Household Nodes
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
house_df=house_df[:50]
'''
Household Feautres Generation
'''
house_df_with_features = generate_household_features(house_df)

def create_standard_id_mapping(house_df):
    """Create index mapping consistent with data_ruxiao_process.py"""
    all_household_ids = set()
    all_household_ids.update(house_df['home'].unique())  # Use 'home' column
    
    household_ids_sorted = sorted(list(all_household_ids))
    
    geohash_to_final_id = {old_id: new_id for new_id, old_id in enumerate(household_ids_sorted, 1)}
    geohash_to_model_idx = {old_id: new_id-1 for old_id, new_id in geohash_to_final_id.items()}
    
    print(f"Created mapping for {len(household_ids_sorted)} households")
    print(f"Sample mapping: {list(geohash_to_model_idx.items())[:3]}")
    
    return geohash_to_final_id, geohash_to_model_idx

# Create mapping BEFORE any renaming
geohash_to_final_id, geohash_to_model_idx = create_standard_id_mapping(house_df_with_features)

# Add processed ID column but keep original 'home' column
house_df_with_features['household_id'] = house_df_with_features['home'].map(geohash_to_final_id)


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

# # k: state dimension (0=repair, 1=vacancy, 2=sales)
# # t: timestep
# t = 0
# k = 0
# ## Compute p_self^k_i(t)
# p_self_series = compute_p_self(house_df_with_features.set_index('home'), house_states, t, k, L=L)
#
# ## Compute p_(ij)^k(t)
# p_ij_series = compute_p_ji_linear(initial_links_df, house_df_with_features, house_states,t, k,L=L)
#
# ### Compute p_(ij)^k(t)
# house_states=update_full_states_one_step(house_states, p_self_series,p_ij_series, initial_links_df,0, 0)
#
# ## Update links_df when t>0
# new_link_df=update_link_matrix_one_step(similarity_df, interaction_df, initial_links_df,house_states,0,alpha_bonding=alpha,beta_form=beta,gamma=gamma)

# ---- helper: compute similarity / interaction for new t if features vary ----

# ---- store link matrices for each t (t index aligned with "time") ----------
link_snapshots = {0: initial_links_df.copy()}
inter_t_list = []
# ---------------- main simulation loop (t = 0 … T-1) -----------------------
generator_probability_log = [] 

for t in tqdm(range(T - 1)):
    print(f'--- sim step  {t}  →  {t+1} ---')

    for k, k_col in enumerate(state_dims):
        print(f"Processing state dimension: {k_col} (k={k})")

        # Compute probabilities (your existing code)
        p_self = compute_p_self(
            house_df_with_features.set_index('home'),
            house_states,
            t=t/T,
            k=k,
            L=L
        )

        p_ji = compute_p_ji_linear(
            link_snapshots[t],
            house_df_with_features,
            house_states,
            t=t/T,
            k=k,
            L=L
        )

        # print(f"p_ji mean: {p_ji.mean()}, p_self mean: {p_self.mean()}")
        # print(f"p_ji shape: {p_ji.shape}, p_self shape: {p_self.shape}")
        # print(f"p_ji max: {p_ji.max()}, p_self max: {p_self.max()}")

        # NEW: Log detailed probabilities before using them
        detailed_entries = log_generator_probabilities(
            p_self, p_ji, link_snapshots[t], house_states, t, k, 
            house_df_with_features, geohash_to_model_idx  # Pass correct mapping
        )
        generator_probability_log.extend(detailed_entries)

        # Continue with existing state update code
        house_states = update_full_states_one_step(
            house_df_with_features,
            house_states,
            p_self,
            p_ji,
            link_snapshots[t],
            t=t,
            k=k
        )

    # Continue with existing link transition code
    sim_t = compute_similarity(house_df_with_features)
    inter_t = compute_interaction_potential(house_df_with_features, house_states, t=t)
    inter_t_list.append(inter_t.copy())

    link_next = update_link_matrix_one_step(
        sim_t,
        inter_t,
        link_snapshots[t],
        house_states,
        t=t,
        alpha_bonding=alpha,
        beta_form=beta,
        gamma=gamma
    )

    link_snapshots[t + 1] = link_next

# ---------------------------------------------------------------------------
# after loop:  merge link snapshots & export results
# ---------------------------------------------------------------------------

# 1. flatten link matrices into long format
link_records = []
for tt, g_df in link_snapshots.items():
    g_upper = g_df.where(np.triu(np.ones(g_df.shape), k=1).astype(bool))
    g_long  = g_upper.stack().reset_index()
    g_long.columns = ['home_i', 'home_j', 'link_type']
    g_long['time'] = tt
    link_records.append(g_long)

links_long_df = pd.concat(link_records, ignore_index=True)

# 2. house_states already accumulated; ensure ordering
house_states = house_states.sort_values(['time', 'home']).reset_index(drop=True)

# ---------------------------------------------------------------------------
#   house_states  –  full T×N×3 node-state table
#   links_long_df –  full (T × |E|) link-type evolution table
# ---------------------------------------------------------------------------

print("Simulation finished.")


'''
Save Results
'''
folder_path = 'sysnthetic_data/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def block_links_per_timestep(df, p):
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Store all indices to be blocked
    to_block_indices = []

    # Loop through each unique timestep
    for t in df['time_step'].unique():
        df_t = df[df['time_step'] == t]
        n_block = int(len(df_t) * p)
        blocked_indices = np.random.choice(df_t.index, size=n_block, replace=False)
        to_block_indices.extend(blocked_indices)

    # Drop the blocked rows
    df_blocked = df.drop(index=to_block_indices)

    return df_blocked

links_long_df=links_long_df.rename(columns={'home_i': 'household_id_1','home_j': 'household_id_2','time': 'time_step'})
house_df_with_features = house_df_with_features.rename(columns={'repair_state':'repair',
                                                                'vacancy_state':'vacancy','sales_state':'sell'})

links_long_df=links_long_df[links_long_df['link_type'] !=0]
blocked_df = block_links_per_timestep(links_long_df, p=p_block)
house_df['latitude'], house_df['longitude'] = zip(*house_df['home'].map(pgh.decode))
house_df = house_df.rename(columns={'home': 'household_id'})
house_states.to_csv(folder_path+'household_states_raw_with_log2.csv',index=False)
links_long_df.to_csv(folder_path+'ground_truth_network_raw_with_log2.csv',index=False)
blocked_df.to_csv(folder_path+'observed_network_raw_with_log2.csv',index=False)
house_df.to_csv(folder_path+'household_locations_raw_with_log2.csv',index=False)
house_df_with_features.to_csv(folder_path+'household_features_raw_with_log2.csv', index=False)
np.save(folder_path+"inter_t_all_raw_with_log2.npy", np.array(inter_t_list))  # shape: (T-1, N, N)
similarity_df.to_csv(folder_path+"similarity_df_raw_with_log2.csv")

print("house_states shape :", house_states.shape)
print("links_long_df shape:", links_long_df.shape)
print("links_long_df shape:", blocked_df.shape)

import pickle
with open(folder_path + 'detailed_generator_probabilities.pkl', 'wb') as f:
    pickle.dump({
        'detailed_log': generator_probability_log,
        'household_order_geohash': house_df_with_features['home'].tolist(),  # Original geohash order
        'household_order_processed': house_df_with_features['household_id'].tolist(),  # Processed ID order
        'household_to_index': geohash_to_model_idx,  # geohash -> 0-based model index
        'geohash_to_final_id': geohash_to_final_id,  # geohash -> 1-based processed ID
        'metadata': {
            'total_timesteps': T,
            'decision_types': state_dims,
            'L': L
        }
    }, f)

print(f"Saved {len(generator_probability_log)} detailed probability entries")
print(f"Saved mapping for {len(geohash_to_model_idx)} households")
