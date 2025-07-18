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
warnings.filterwarnings("ignore")


# Hyper parameter definition
alpha=0.5
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

# ---- store p_self and p_ji statistics for each time step ----
p_self_stats = []  # Will store records of {t, k, p_self_mean}
p_ji_stats = []    # Will store records of {t, k, p_ji_nonzero_mean}

# ---------------- main simulation loop (t = 0 … T-1) -----------------------
for t in tqdm(range(T - 1)):              # we already have states at t, produce t+1
    print(f'--- sim step  {t}  →  {t+1} ---')

    # -------------------------------------------------
    # state-update for each dimension k ∈ {0,1,2}
    # -------------------------------------------------
    for k, k_col in enumerate(state_dims):

        # --- compute p_self and p_ji for current (t, k) ---------------------
        p_self = compute_p_self(
            house_df_with_features.set_index('home'),
            house_states,
            t=t/T,
            k=k,
            L=L
        )

        p_ji = compute_p_ji_linear(
            link_snapshots[t],                # links at time t
            house_df_with_features,
            house_states,
            t=t/T,
            k=k,
            L=L
        )
        
        # ---- Record statistics for p_self and p_ji ----
        # Record p_self mean
        p_self_mean = p_self.mean()
        p_self_stats.append({
            'time_step': t,
            'dimension': k,
            'dimension_name': state_dims[k],
            'p_self_mean': p_self_mean
        })
        
        # Record p_ji non-zero mean
        p_ji_values = p_ji.values
        nonzero_mask = p_ji_values != 0
        if nonzero_mask.any():
            p_ji_nonzero_mean = p_ji_values[nonzero_mask].mean()
        else:
            p_ji_nonzero_mean = 0.0
        
        p_ji_stats.append({
            'time_step': t,
            'dimension': k,
            'dimension_name': state_dims[k],
            'p_ji_nonzero_mean': p_ji_nonzero_mean,
            'nonzero_count': nonzero_mask.sum()
        })
        
        # --- update states (writes into time t+1 row) -----------------------
        house_states = update_full_states_one_step(
            house_df_with_features,
            house_states,                     # full long table
            p_self,
            p_ji,
            link_snapshots[t],                # links at t
            t=t,
            k=k
        )

    # -------------------------------------------------
    # link-transition   G_t  →  G_{t+1}
    # -------------------------------------------------
    # similarity / interaction may be time-varying; here we fetch for step t
    sim_t = compute_similarity(house_df_with_features)
    inter_t = compute_interaction_potential(house_df_with_features, house_states, t=t+1)
    inter_t_list.append(inter_t.copy())

    link_next = update_link_matrix_one_step(
        sim_t,
        inter_t,
        link_snapshots[t],                    # G_t
        house_states,
        t=t,
        alpha_bonding=alpha,
        beta_form=beta,
        gamma=gamma
    )

    link_snapshots[t + 1] = link_next        # store snapshot for next step

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
count_ones = (initial_links_df == 1).sum().sum()
print(f"number of bonding links: {count_ones/2}")
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
house_df_with_features = house_df_with_features.rename(columns={'home': 'household_id', 'repair_state':'repair',
                                                                'vacancy_state':'vacancy','sales_state':'sell'})

links_long_df=links_long_df[links_long_df['link_type'] !=0]
blocked_df = block_links_per_timestep(links_long_df, p=p_block)
house_df['latitude'], house_df['longitude'] = zip(*house_df['home'].map(pgh.decode))
house_df = house_df.rename(columns={'home': 'household_id'})
house_states.to_csv(folder_path+'household_states_raw.csv',index=False)
links_long_df.to_csv(folder_path+'ground_truth_network_raw.csv',index=False)
blocked_df.to_csv(folder_path+'observed_network_raw.csv',index=False)
house_df.to_csv(folder_path+'household_locations_raw.csv',index=False)
house_df_with_features.to_csv(folder_path+'household_features_raw.csv', index=False)
np.save(folder_path+"inter_t_all.npy", np.array(inter_t_list))  # shape: (T-1, N, N)
similarity_df.to_csv(folder_path+"similarity_df_raw.csv")

# ---- Save p_self and p_ji statistics ----
# Convert statistics lists to DataFrames
p_self_df = pd.DataFrame(p_self_stats)
p_ji_df = pd.DataFrame(p_ji_stats)

# Save statistics to CSV files
p_self_df.to_csv(folder_path+"p_self_statistics.csv", index=False)
p_ji_df.to_csv(folder_path+"p_ji_statistics.csv", index=False)


