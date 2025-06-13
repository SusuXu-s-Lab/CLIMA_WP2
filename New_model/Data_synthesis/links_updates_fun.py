import numpy as np
from scipy.special import softmax
import pandas as pd

def generate_initial_links(similarity_df, interaction_df, alpha_0=0.9, beta_0=0.5):
    homes = similarity_df.index.tolist()
    N = len(homes)

    # Get upper triangle indices (i < j)
    triu_indices = np.triu_indices(N, k=1)
    sim_vals = similarity_df.values[triu_indices]
    inter_vals = interaction_df.values[triu_indices]

    # Compute logits for each pair (3 logits: for no-link, bonding, bridging)
    logit_0 = np.ones_like(sim_vals)  # base score for "no link"
    logit_1 = alpha_0 * sim_vals      # bonding
    logit_2 = beta_0 * inter_vals     # bridging

    logits = np.vstack((logit_0, logit_1, logit_2)).T  # shape (K, 3)
    probs = softmax(logits, axis=1)  # shape (K, 3)

    # Sample link type from categorical distribution
    link_types = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])

    # Fill symmetric link matrix
    link_matrix = np.zeros((N, N), dtype=int)
    link_matrix[triu_indices] = link_types
    link_matrix[(triu_indices[1], triu_indices[0])] = link_types

    return pd.DataFrame(link_matrix, index=homes, columns=homes)