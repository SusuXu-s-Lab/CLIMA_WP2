import pdb

import numpy as np
from scipy.special import softmax
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import pygeohash as pgh
from config import ScenarioConfig, ALL_SCENARIOS, get_scenario

# ----------------------------------------------------------------------
# Build symmetric Top-K nearest-neighbor candidate mask by geographic distance
# ----------------------------------------------------------------------
def geo_topk_mask(latlon_deg: np.ndarray, top_k: int, hard_cut_m: float = 0.0):
    """
    Build a symmetric Top-K nearest-neighbor mask by geographic distance.
    Args:
        latlon_deg: np.ndarray of shape (N, 2), columns are [lat(deg), lon(deg)]
        top_k: each node keeps only its K nearest neighbors as edge candidates
        hard_cut_m: if > 0, drop pairs whose great-circle distance exceeds this (meters)
    Returns:
        cand_mask: (N, N) boolean matrix, symmetric, diag=False
        dist_m:    (N, N) float matrix of great-circle distances in meters
    """
    # Convert degrees to radians for haversine_distances
    coords_rad = np.radians(latlon_deg.astype(float))
    # haversine_distances returns distances in radians; convert to meters
    dist_m = haversine_distances(coords_rad) * 6_371_000.0

    N = dist_m.shape[0]
    cand_mask = np.zeros((N, N), dtype=bool)

    # For each node, keep only its top-K nearest neighbors (exclude self)
    for i in range(N):
        order = np.argsort(dist_m[i])      # ascending by distance
        nn = order[1:1 + min(top_k, N - 1)]
        cand_mask[i, nn] = True

    # Symmetrize so that i-j is allowed if either i picked j or j picked i
    cand_mask |= cand_mask.T

    # Remove self
    np.fill_diagonal(cand_mask, False)

    # Optional hard cutoff in meters
    if hard_cut_m and hard_cut_m > 0.0:
        cand_mask &= (dist_m <= hard_cut_m)

    return cand_mask, dist_m


# ----------------------------------------------------------------------
# Helper to apply the candidate mask on an "upper-triangle vectorized" p_edge
# ----------------------------------------------------------------------
def apply_geo_topk_filter_to_uppertri(p_edge_vec: np.ndarray,
                                      cand_mask: np.ndarray,
                                      iu: tuple):
    """
    Apply candidate mask to an upper-triangle vectorized edge-probability array.
    Args:
        p_edge_vec: 1-D array of edge probabilities corresponding to (i<j) pairs
        cand_mask:  (N, N) boolean matrix of allowed pairs
        iu:         tuple of (rows, cols) indices from np.triu_indices(N, k=1)
    Returns:
        p_edge_filtered: 1-D array with non-candidate pairs forced to 0
        cand_upper_vec:  boolean vector mask aligned with p_edge_vec
    """
    cand_upper_vec = cand_mask[iu]
    p_edge_filtered = p_edge_vec.copy()
    p_edge_filtered[~cand_upper_vec] = 0.0
    return p_edge_filtered, cand_upper_vec

def generate_initial_links(similarity_df, interaction_df, config: ScenarioConfig,
                           alpha_0=0.9, beta_0=0.5, epsilon=1e-8, output_folder=None):
    """
    Two-stage link generation (density -> type) with GEO-aware calibration.

    Stage 1 (density): decide whether an undirected edge exists between i<j
        p_edge = sigmoid(b_edge + a1 * qnorm(sim) + a2 * qnorm(inter))
        b_edge is calibrated by bisection to hit a target average degree,
        BUT NOW only counting pairs inside the GEO Top-K candidate mask.

    Stage 2 (type): only for existing edges, decide {bonding=1, bridging=2}
        Softmax over linear logits:
            logit_bond   = c_bond + alpha_0 * qnorm(sim)
            logit_bridge = c_bridge + beta_0  * qnorm(inter)

    Returns:
        pd.DataFrame (N x N) with {0: no link, 1: bonding, 2: bridging}
    """
    import os
    import numpy as np
    import pandas as pd
    import pygeohash as pgh  # used to decode geohash -> (lat, lon)

    # ---------- helpers ----------
    def qnorm(vec: np.ndarray) -> np.ndarray:
        """Rank-based quantile normalization to [0,1] (stable, monotonic)."""
        order = np.argsort(vec)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(vec), dtype=float) + 0.5
        return ranks / len(vec)

    # homes & shape
    homes = similarity_df.index.tolist()
    N = len(homes)
    iu = np.triu_indices(N, k=1)  # upper triangle (i<j)

    # flatten features
    sim_vals   = similarity_df.values[iu]
    inter_vals = interaction_df.values[iu]
    s_sim   = qnorm(sim_vals)
    s_inter = qnorm(inter_vals)

    # ---------- GEO Top-K candidate mask (built ONCE, re-used in calibration & sampling) ----------
    # build lat/lon array aligned with similarity_df order
    latlon = np.array([pgh.decode(h) for h in similarity_df.index.tolist()])  # [(lat, lon), ...]
    # Expect geo_topk_mask(latlon_deg, top_k, hard_cut_m) is already defined in this file.
    cand_mask, _dist_m = geo_topk_mask(
        latlon_deg=latlon,
        top_k=getattr(config, 'top_k_neighbors', 100),
        hard_cut_m=getattr(config, 'geo_hard_cutoff', 0.0)  # meters; 0 disables cutoff
    )
    cand_upper = cand_mask[iu]  # boolean vector aligned with upper-triangle vectors

    # ---------- Stage 1: GEO-aware b_edge calibration ----------
    target_avg_degree = float(getattr(config, 'target_avg_degree', 3.0))
    print(f"Calibrating b_edge to hit target average degree (geo-aware): {target_avg_degree}")
    target_edges = target_avg_degree * N / 2.0

    # Use alpha_0, beta_0 as linear weights; b_edge absorbs global level
    a1 = float(alpha_0)
    a2 = float(beta_0)

    # Expected average degree under b with GEO candidate mask
    def avg_degree_given(b: float) -> float:
        logits_tmp = b + a1 * s_sim + a2 * s_inter        # shape: (#upper)
        p_tmp = 1.0 / (1.0 + np.exp(-logits_tmp))
        # Only count GEO-candidate pairs
        p_tmp = p_tmp * cand_upper.astype(float)
        return 2.0 * p_tmp.sum() / N

    # bisection on b_edge
    lo, hi = -15.0, 15.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if avg_degree_given(mid) < target_avg_degree:
            lo = mid
        else:
            hi = mid
    b_edge = 0.5 * (lo + hi)
    # compute final p_edge (still vector over upper-triangle)
    logits_edge = b_edge + a1 * s_sim + a2 * s_inter
    p_edge = 1.0 / (1.0 + np.exp(-logits_edge))
    # enforce GEO candidate mask at sampling time
    p_edge[~cand_upper] = 0.0

    # (optional) quick sanity print
    realized_deg_est = 2.0 * p_edge.sum() / N
    print(f"[geo-aware calib] target={target_avg_degree:.2f}, expected≈{realized_deg_est:.2f}")

    # sample existence
    edge_mask = (np.random.random(len(p_edge)) < p_edge)

    # ---------- Stage 2: Type decision on existing edges ----------
    # keep bridging slightly dominant by biasing bridge
    c_bond, c_bridge = 0.0, 0.3
    logit_bond   = c_bond   + alpha_0 * s_sim
    logit_bridge = c_bridge + beta_0  * s_inter
    logits_2 = np.stack([logit_bond, logit_bridge], axis=1)
    logits_2 = logits_2 - logits_2.max(axis=1, keepdims=True)
    probs_2  = np.exp(logits_2)
    probs_2  = probs_2 / probs_2.sum(axis=1, keepdims=True)  # [:,0]=p_bond, [:,1]=p_bridge

    link_types_vec = np.zeros(len(p_edge), dtype=int)  # 0=no link
    idx = np.where(edge_mask)[0]
    draws = np.random.random(len(idx))
    link_types_vec[idx] = (draws < probs_2[idx, 1]).astype(int) + 1  # 1=bond, 2=bridge

    # ---------- build symmetric matrix ----------
    link_matrix = np.zeros((N, N), dtype=int)
    link_matrix[iu] = link_types_vec
    link_matrix[(iu[1], iu[0])] = link_types_vec

    # ---------- optional export for diagnostics ----------
    try:
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            out_path = os.path.join(output_folder, "link_probs_raw.csv")
            # expectation at t=0
            P_no      = 1.0 - p_edge
            P_bridge  = probs_2[:, 1]
            P_bond    = 1.0 - P_bridge
            df_out = pd.DataFrame({
                "P_no_link": P_no,
                "P_bonding": p_edge * P_bond,
                "P_bridging": p_edge * P_bridge
            })
            df_out.to_csv(out_path, index=False)
            print(f"Link probabilities saved to: {out_path}")
    except Exception as e:
        # fail silently if the path is restricted, but show a hint
        print(f"[warn] could not write link_probs_raw.csv: {e}")

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
    state_cols = ['vacancy_state','repair_state','sales_state']
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
    # Original constant weights (commented out):
    # w_static = np.array([0.1, 0.2, 0.2])             # income, age, race
    # w_static = np.array([-2.7, -4.8, -3.3])*3            
    # w_hist   = np.full(2 * L, -0.05)                  # L steps of 2 non-k dims
    # w_time = np.array([-0.05])
    
    # Sampled weights from distributions with small variance:
    w_static = np.random.normal(np.array([-2.7, -4.8, -3.3])*3, 0.1)  # income, age, race weights
    w_hist   = np.random.normal(-0.05, 0.005, 2 * L)    # L steps of 2 non-k dims, small variance
    w_time   = np.random.normal([-0.05], 0.005, 1)      # time weight, small variance
    weights  = np.concatenate([w_static, w_hist, w_time])
    
    assert weights.shape[0] == all_feat.shape[1], "Shape mismatch in feature × weight"

    # (5) sigmoid scoring with time-based cyclical variation
    dot = all_feat @ weights
    
    # # Add periodic/seasonal variation based on time
    # # Multiple cycles: annual (period=12), quarterly (period=3), monthly fluctuation
    # annual_cycle = 0.1 * np.sin(2 * np.pi * t / 12.0)      # Annual cycle with amplitude 0.1
    # quarterly_cycle = 0.1 * np.cos(2 * np.pi * t / 3.0)   # Quarterly cycle with amplitude 0.05
    # monthly_trend = 0.05 * np.sin(2 * np.pi * t / 1.0)     # Monthly fluctuation with amplitude 0.02
    
    # # Combine cyclical components
    # time_variation = annual_cycle + quarterly_cycle + monthly_trend
    
    # # Apply time variation to the dot product
    # dot_with_cycles = dot + time_variation
    dot_with_cycles = dot
    
    p_self = 1 / (1 + np.exp(-dot_with_cycles))
    
    # Cap probabilities: limit to maximum 0.8 and set values below 0.1 to 0
    p_self = np.minimum(p_self, 0.6)  # Cap at 0.8
    p_self[p_self < 0.1] = 0.0        # Set values below 0.1 to 0
    
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

    hist_src = np.broadcast_to(hist_flat[:, None, :],  # src=j，
                               (N, N, 2 * L))  # => (N,N,2L)
    hist_tgt = np.broadcast_to(hist_flat[None, :, :],  # tgt=i，
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

    # Standardization
    D = feat_all.shape[-1]
    feat_all_norm = np.empty_like(feat_all)

    for d in range(D):
        x = feat_all[..., d]
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        if np.isclose(x_max, x_min):  
            feat_all_norm[..., d] = 0.0
        else:
            feat_all_norm[..., d] = (x - x_min) / (x_max - x_min)


    # ---------- (4) weighted linear score ----------
    # Original constant weights (commented out):
    # w_demo   = np.array([-4.7, -4.5, -4.7])
    # w_hist   = np.full(2 * L, -0.04)                # applies to both src & tgt
    # w_link   = np.array([1.2])
    # w_dist   = np.array([-1])
    # w_time = np.array([-0.05])
    
    # Sampled weights from distributions with small variance:
    w_demo   = np.random.normal([-4.7, -4.5, -4.7], 0.05, 3)  # demographic weights
    w_hist   = np.random.normal(-0.04, 0.005, 2 * L)           # history weights, smaller variance
    w_link   = np.random.normal([1.2], 0.05, 1)                # link weight
    w_dist   = np.random.normal([-1], 0.05, 1)                 # distance weight
    w_time   = np.random.normal([-0.2], 0.005, 1)             # time weight, smaller variance
    weights  = np.concatenate([w_demo, w_hist, w_hist, w_link, w_dist, w_time])  # (3+4L+2,)
    scores = np.tensordot(feat_all_norm, weights, axes=([-1], [0]))            # (N,N)
    p_mat  = 1.0 / (1.0 + np.exp(-scores))                                 # sigmoid
    
    # Limit values to maximum 0.5 and set values below 0.1 to 0
    p_mat = np.minimum(p_mat, 0.5)  # Cap at 0.5
    p_mat[p_mat < 0.1] = 0.0        # Set values below 0.1 to 0
    
    # p_mat  = 0.9 / (1.0 + np.exp(-scores))                                 # sigmoid


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

    for i in range(N):
        for j in range(i + 1, N):                  # upper-triangle only
            prev = link_prev[i, j]

            # 1)  from NO-LINK  (Eq. 13-14) ---------------------------------
            if prev == 0:
                # No new bridging links are created - stay as no-link
                new = 0

            # 2)  from BONDING  (Eq. 15)  – stays bonding
            elif prev == 1:
                new = 1                        # p11 = 1

            # 3)  from BRIDGING  (Eq.16-17)
            else:                                  # prev == 2
                both_stay = (vac_t[i] == 0) and (vac_t[j] == 0)
                # Bridging links only disappear if either household becomes vacant
                # They are not affected by interaction potential
                new = 2 if both_stay else 0

            # fill symmetric entries
            link_new[i, j] = link_new[j, i] = new

    return pd.DataFrame(link_new, index=homes, columns=homes)