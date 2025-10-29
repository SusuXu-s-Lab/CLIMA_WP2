import torch

def get_state_history_excluding_k(household_idx, decision_k, states, time, L):
    """
    OPTIMIZED: Get state history s_i(t:t-L+1)^{-k} excluding decision type k.
    Enhanced with vectorized operations for better performance.
    """
    if isinstance(household_idx, torch.Tensor):
        if household_idx.numel() == 0:
            return torch.empty(0, L * 2)
        household_idx_list = household_idx.tolist()
    else:
        household_idx_list = household_idx if isinstance(household_idx, list) else [household_idx]
    
    if len(household_idx_list) == 0:
        return torch.empty(0, L * 2)
    
    batch_size = len(household_idx_list)
    start_time = max(0, time - L + 1)
    end_time = min(time + 1, states.shape[1])
    
    # Vectorized extraction of state histories
    # OPTIMIZATION: Detach since these are observed historical states (no gradient needed)
    household_indices = torch.tensor(household_idx_list, dtype=torch.long)
    state_hist = states[household_indices, start_time:end_time, :].detach()  # [batch_size, time_steps, 3]
    
    # Remove decision_k dimension vectorially
    other_decisions = [i for i in range(3) if i != decision_k]
    state_hist_excluding_k = state_hist[:, :, other_decisions]  # [batch_size, time_steps, 2]
    
    # Calculate expected length
    expected_length = L * 2  # L timesteps × 2 decision types (excluding k)
    actual_length = state_hist_excluding_k.shape[1] * state_hist_excluding_k.shape[2]
    
    # Reshape to flat
    state_hist_flat = state_hist_excluding_k.view(batch_size, -1)
    
    # Pad with zeros if necessary (for early timesteps)
    if actual_length < expected_length:
        padding_length = expected_length - actual_length
        padding = torch.zeros(batch_size, padding_length, 
                            dtype=state_hist.dtype, device=state_hist.device)
        state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
    
    return state_hist_flat



def get_full_state_history(household_idx, states, time, L):
    """
    OPTIMIZED: Get full state history S_i(t:t-L+1) including all decision types.
    Enhanced with vectorized operations for better performance.
    
    Historical states are detached to reduce computational graph complexity.
    """
    if isinstance(household_idx, torch.Tensor):
        if household_idx.numel() == 0:
            return torch.empty(0, L * 3)
        household_idx_list = household_idx.tolist()
    else:
        household_idx_list = household_idx if isinstance(household_idx, list) else [household_idx]
    
    if len(household_idx_list) == 0:
        return torch.empty(0, L * 3)
    
    batch_size = len(household_idx_list)
    start_time = max(0, time - L + 1)
    end_time = min(time + 1, states.shape[1])
    
    # Vectorized extraction of state histories
    # OPTIMIZATION: Detach since these are observed historical states (no gradient needed)
    household_indices = torch.tensor(household_idx_list, dtype=torch.long)
    state_hist = states[household_indices, start_time:end_time, :].detach()  # [batch_size, time_steps, 3]
    
    # Calculate expected length
    expected_length = L * 3  # L timesteps × 3 decision types
    actual_length = state_hist.shape[1] * state_hist.shape[2]
    
    # Reshape to flat
    state_hist_flat = state_hist.view(batch_size, -1)
    
    # Pad with zeros if necessary (for early timesteps)
    if actual_length < expected_length:
        padding_length = expected_length - actual_length
        padding = torch.zeros(batch_size, padding_length, 
                            dtype=state_hist.dtype, device=state_hist.device)
        state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
    
    return state_hist_flat


def compute_pairwise_features(features_i, features_j):
    """Compute pairwise features f_ij = |features_i - features_j|"""
    return torch.abs(features_i - features_j)


def build_neighbor_index_from_distances(distances: torch.Tensor,
                                        radius: float = None,
                                        top_k: int = None):
    """
    Build a per-node candidate neighbor list using an N x N distance matrix.

    Args:
        distances: [N, N] symmetric matrix (0 on diagonal), same device you train on.
        radius:   keep j if distances[i,j] <= radius (optional)
        top_k:    keep at most K nearest neighbors per node (optional)

    Returns:
        neighbor_index: List[List[int]] where neighbor_index[i] is a list of node ids (ints)
                        considered as candidates for i.
    """
    N = distances.shape[0]
    device = distances.device
    neighbor_index = [[] for _ in range(N)]

    # avoid self as neighbor by adding a huge number to the diagonal
    D = distances.clone()
    D.fill_diagonal_(float('inf'))

    for i in range(N):
        d_i = D[i]  # [N]

        # radius filter
        if radius is not None:
            keep = d_i <= radius
            idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        else:
            idx = torch.arange(N, device=device)

        # top-k filter (on the *already* radius-filtered subset)
        if top_k is not None and idx.numel() > top_k:
            vals, order = torch.topk(d_i[idx], k=top_k, largest=False)
            idx = idx[order]

        neighbor_index[i] = idx.tolist()

    # (optional) symmetrize so that if j is in i’s list, i is also in j’s list
    for i in range(N):
        for j in neighbor_index[i]:
            if i not in neighbor_index[j]:
                neighbor_index[j].append(i)

    return neighbor_index