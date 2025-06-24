import pdb

import numpy as np
from scipy.special import softmax
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import pygeohash as pgh

def generate_initial_links(similarity_df, interaction_df, alpha_0=0.9, beta_0=0.5, epsilon=1e-8):
    homes = similarity_df.index.tolist()
    N = len(homes)

    # Get upper triangle indices (i < j)
    triu_indices = np.triu_indices(N, k=1)
    sim_vals = similarity_df.values[triu_indices]
    inter_vals = interaction_df.values[triu_indices]
    # Compute logits for each pair (3 logits: for no-link, bonding, bridging)
    logit_0 = np.zeros_like(sim_vals)  # base score for "no link"
    logit_1 = np.log(alpha_0 * sim_vals + epsilon)  # bonding link
    logit_2 = np.log(inter_vals + epsilon)  # bridging link

    logits = np.vstack((logit_0, logit_1, logit_2)).T  # shape (K, 3)
    probs = softmax(logits, axis=1)  # shape (K, 3)
    # Sample link type from categorical distribution
    link_types = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])

    # Fill symmetric link matrix
    link_matrix = np.zeros((N, N), dtype=int)
    link_matrix[triu_indices] = link_types
    link_matrix[(triu_indices[1], triu_indices[0])] = link_types

    output_path = "link_probs.csv"
    df = pd.DataFrame(probs, columns=["P_no_link", "P_bonding", "P_bridging"])
    df.to_csv(output_path, index=False)
    return pd.DataFrame(link_matrix, index=homes, columns=homes)

def compute_p_self(house_df, full_state_df, t, k, L=3):
    """
    Compute p_self^k_i(t) using static features + full unfolded state history.

    Args:
        house_df: DataFrame with static features, indexed by 'home'
        full_state_df: DataFrame with all states across time
        t: Current time step
        k: Decision dimension (0=repair, 1=vacancy, 2=sales)
        L: History window length

    Returns:
        Series of p_self_i^k(t), indexed by home ID
    """
    state_cols = ['repair_state', 'vacancy_state', 'sales_state']
    non_k_cols = [col for i, col in enumerate(state_cols) if i != k]  # length 2
    homes = house_df.index.tolist()
    N = len(homes)
    idx_map = {h: i for i, h in enumerate(homes)}

    # (1) static features
    static_feat = house_df[['income', 'age', 'race']].copy()  # (N, 3)

    # (2) collect past L-step non-k states
    hist_tensor = np.zeros((N, L, len(non_k_cols)))  # shape (N, L, 2)

    for offset in range(1, L + 1):
        t_lookup = t - offset
        if t_lookup < 0:
            continue
        df_slice = (full_state_df[full_state_df['time'] == t_lookup]
                    .set_index('home')[non_k_cols])
        for h in df_slice.index.intersection(homes):
            hist_tensor[idx_map[h], L - offset, :] = df_slice.loc[h].values

    hist_flat = hist_tensor.reshape(N, -1)  # shape (N, 2L)

    # (3) concatenate features: [income, age, race, s^{-k}_{t-L:t-1}, time]
    time_feat = np.full((N, 1), t)
    all_feat = np.concatenate([
        static_feat.values,
        hist_flat,
        time_feat
    ], axis=1)  # final shape = (N, 3 + 2L + 1)

    # (4) weights
    w_static = np.array([1.0, 0.5, 0.5])             # income, age, race
    w_hist   = np.full(2 * L, -0.8)                  # L steps of 2 non-k dims
    w_time = np.array([-15])
    weights  = np.concatenate([w_static, w_hist, w_time])/50

    assert weights.shape[0] == all_feat.shape[1], "Shape mismatch in feature × weight"

    # (5) sigmoid scoring
    dot = all_feat @ weights
    p_self = 1 / (1 + np.exp(-dot))
    return pd.Series(p_self, index=house_df.index)

def compute_p_ji_linear(link_df,
                        house_df_with_features,
                        full_state_df,
                        t: float,
                        k: int,
                        L: int = 3):
    """
    Same signature / semantics, but vectorised for speed.
    """
    homes   = house_df_with_features['home'].tolist()
    N       = len(homes)
    idx_map = {h: i for i, h in enumerate(homes)}

    # ---------- (1) static feature blocks ----------
    demo_mat = house_df_with_features[['income', 'age', 'race']].values     # (N,3)
    f_demo   = np.abs(demo_mat[:, None, :] - demo_mat[None, :, :])          # (N,N,3)

    coords     = house_df_with_features['home'].apply(pgh.decode).tolist()
    geo_dist   = haversine_distances(np.radians(coords)) * 6_371_000        # (N,N)

    # ---------- (2) history tensor ----------
    state_cols  = ['repair_state', 'vacancy_state', 'sales_state']
    non_k_cols  = [c for i_, c in enumerate(state_cols) if i_ != k]         # length 2
    hist_tensor = np.zeros((N, L, len(non_k_cols)))                         # (N,L,2)

    for offset in range(1, L + 1):
        t_lkp = t - offset
        if t_lkp < 0:
            break
        slice_df = (full_state_df[full_state_df['time'] == t_lkp]
                    .set_index('home')[non_k_cols])
        common = slice_df.index.intersection(homes)
        hist_tensor[[idx_map[h] for h in common], L - offset, :] = slice_df.loc[common].values

    hist_flat = hist_tensor.reshape(N, -1)                                  # (N,2L)

    # ---------- (3) assemble features for all (j,i) ----------
    link_mat = link_df.values  # (N,N)
    link_feat = link_mat[..., None]  # (N,N,1)

    hist_src = np.broadcast_to(hist_flat[:, None, :],  # src=j，行向量
                               (N, N, 2 * L))  # => (N,N,2L)
    hist_tgt = np.broadcast_to(hist_flat[None, :, :],  # tgt=i，列向量
                               (N, N, 2 * L))  # => (N,N,2L)
    time_feat = np.full((N, N, 1), t)   # (N,N,1)

    feat_all = np.concatenate(
        [f_demo,  # (N,N,3)
         hist_src,  # (N,N,2L)
         hist_tgt,  # (N,N,2L)
         link_feat,  # (N,N,1)
         geo_dist[..., None],
         time_feat],  # (N,N,1)
        axis=-1)  # -> (N,N,3+4L+2)

    # ---------- (4) weighted linear score ----------
    w_demo   = np.array([-2.0, -1.5, -1.5])
    w_hist   = np.full(2 * L, -1.0)                # applies to both src & tgt
    w_link   = np.array([0.8])
    w_dist   = np.array([-4])
    w_time = np.array([-15.0])
    weights  = np.concatenate([w_demo, w_hist, w_hist, w_link, w_dist, w_time])/50     # (3+4L+2,)
    scores = np.tensordot(feat_all, weights, axes=([-1], [0]))             # (N,N)
    p_mat  = 1.0 / (1.0 + np.exp(-scores))                                 # sigmoid

    # ---------- (5) mask out self-pairs & absent links ----------
    mask = (link_mat == 0) | np.eye(N, dtype=bool)
    p_mat[mask] = 0.0

    return pd.DataFrame(p_mat, index=homes, columns=homes)

def update_link_matrix_one_step(similarity_df: pd.DataFrame,
                                interaction_df: pd.DataFrame,
                                links_prev_df: pd.DataFrame,
                                house_states: pd.DataFrame,
                                t: int,
                                alpha_bonding: float = 0.9,
                                beta_form: float   = 0.5,
                                gamma: float       = 0.3) -> pd.DataFrame:
    """
    One-step link–transition update  ( t-1  ➜  t )  following Eq.(13)–(17).

    Parameters
    ----------
    similarity_df   : DataFrame  –  similarity(i,j,t)  (symmetric, N×N)
    interaction_df  : DataFrame  –  interaction_potential(i,j,t)  (symmetric, N×N)
    links_prev_df   : DataFrame  –  link matrix ℓ_{ij}(t-1)  (values 0/1/2, symmetric)
    house_states    : DataFrame  –  long table with ['home','time','vacancy_state', ...]
    t               : int        –  current time step (links_prev_df is t-1; we create t)
    alpha_bonding   : float      –  coefficient in Eq.(13)
    beta_form       : float      –  coefficient in Eq.(13)
    gamma           : float      –  decay factor for bridging when either household is vacant

    Returns
    -------
    DataFrame  –  new symmetric link matrix  ℓ_{ij}(t)
    """

    homes = similarity_df.index.tolist()          # common ordering for all matrices
    N = len(homes)

    # --- fetch vacancy indicators at time t ---------------------------------
    vac_t = (house_states[house_states['time'] == t]
             .set_index('home')
             .reindex(homes)['vacancy_state']
             .fillna(0)
             .astype(int)
             .values)                              # shape (N,)

    # --- initialise new link matrix with zeros ------------------------------
    link_new = np.zeros((N, N), dtype=int)

    link_prev = links_prev_df.values
    sim_mat = similarity_df.values
    inter_mat = interaction_df.values

    rng = np.random.default_rng()

    for i in range(N):
        for j in range(i + 1, N):                  # upper-triangle only
            prev = link_prev[i, j]

            # 1)  from NO-LINK  (Eq. 13-14) ---------------------------------
            if prev == 0:

                probs = [1.0-inter_mat[i, j], inter_mat[i, j]]
                # print(probs)
                new = rng.choice([0, 2], p=probs)

            # 2)  from BONDING  (Eq. 15)  – stays bonding
            elif prev == 1:
                new = 1                        # p11 = 1

            # 3)  from BRIDGING  (Eq.16-17)
            else:                                  # prev == 2
                both_stay = (vac_t[i] == 0) and (vac_t[j] == 0)
                p22 = inter_mat[i, j] if both_stay else 0
                p22 = np.clip(p22, 0.0, 1.0)
                p20 = 1 - p22                    # may drop to no-link
                probs=[p20, p22]
                # print(probs)
                # pdb.set_trace()
                new = rng.choice([0, 2], p=probs)

            # fill symmetric entries
            link_new[i, j] = link_new[j, i] = new

    return pd.DataFrame(link_new, index=homes, columns=homes)