import pdb

from scipy.spatial.distance import cdist
import pygeohash as pgh
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import pandas as pd

# Example function to compute similarity matrix at time t
def compute_similarity(house_df):
    # Step 1: demographic distance
    demo = house_df[['income', 'age', 'race']].values
    demo_dist = cdist(demo, demo, metric='euclidean')
    sigma_demo = np.median(demo_dist[np.triu_indices(len(demo), k=1)])

    # Step 2 & 3: decode and compute geo distance (vectorized)
    coords = house_df['home'].apply(lambda g: pgh.decode(g))
    coords_rad = np.radians(coords.tolist())
    geo_dist = haversine_distances(coords_rad) * 6371000
    sigma_geo = np.median(geo_dist[np.triu_indices(len(coords_rad), k=1)])

    # Step 4: similarity
    similarity = np.exp(
        -(demo_dist ** 2 / sigma_demo ** 2 + geo_dist ** 2 / sigma_geo ** 2)
    )
    return pd.DataFrame(similarity, index=house_df['home'], columns=house_df['home'])

# Example function to compute interaction potential matrix at time t
def compute_interaction_potential(house_df, state_df, t):
    # Prepare data at time t
    df_t = state_df[state_df['time'] == t].set_index('home')
    house_df = house_df.set_index('home')
    common_homes = house_df.index.intersection(df_t.index)
    house_df = house_df.loc[common_homes]
    df_t = df_t.loc[common_homes]

    # Step 1: compute f_ij(t) = |demo_i - demo_j|
    demo = house_df[['income', 'age', 'race']].values
    f_ij = np.abs(demo[:, None, :] - demo[None, :, :])  # shape (N, N, 3)

    # Step 2: extract s_i and s_j
    s_mat = df_t[['repair_state', 'vacancy_state', 'sales_state']].values  # (N, 3)
    s_i = np.repeat(s_mat[:, None, :], len(s_mat), axis=1)  # (N, N, 3)
    s_j = np.repeat(s_mat[None, :, :], len(s_mat), axis=0)  # (N, N, 3)
    # Step 3: compute distance from geohash
    coords = house_df.index.to_series().apply(lambda g: pgh.decode(g)).tolist()
    coords_rad = np.radians(coords)
    geo_dist = haversine_distances(coords_rad) * 6371000  # in meters
    dist_feat = geo_dist[:, :, None]  # (N, N, 1)
    dist_feat = (dist_feat - dist_feat.min()) / (dist_feat.max() - dist_feat.min() + 1e-8)
    # Step 4: concatenate full feature vector as [f_ij, s_i, s_j, dist]
    full_feat = np.concatenate([f_ij, s_i, s_j, dist_feat], axis=2)  # (N, N, 10)
    if t==0:
        weights = np.array([-3.0, -6.0, -7.0,     # f_ij part
                            -1.0, -3.0, -1.0,     # s_i part
                            -1.0, -5.0, -2.0,     # s_j part
                            -20.0])/15       # dist_ij
    else:
        weights = np.array([-9, -13.0, -15.0,     # f_ij part
                            -10.0, -7.0, -5.0,     # s_i part
                            -6.0, -5.0, -4.0,     # s_j part
                            -20.0])/5     # dist_ij
    dot = np.tensordot(full_feat, weights, axes=([2], [0]))  # shape (N, N)
    # print("dot range:", dot.min(), dot.max(), dot.mean())
    interaction = 1 / (1 + np.exp(-dot))  # sigmoid
    return pd.DataFrame(interaction, index=house_df.index, columns=house_df.index)
