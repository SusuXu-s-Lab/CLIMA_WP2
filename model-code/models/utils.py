import torch

# def get_state_history_excluding_k(household_idx, decision_k, states, time, L):
#     """Get state history s_i(t:t-L+1)^{-k} excluding decision type k."""
#     if isinstance(household_idx, torch.Tensor):
#         household_idx = household_idx.tolist()
    
#     start_time = max(0, time - L + 1)
#     end_time = time + 1
    
#     state_hist = states[household_idx, start_time:end_time, :]
#     other_decisions = [i for i in range(3) if i != decision_k]
#     state_hist_excluding_k = state_hist[:, :, other_decisions]
    
#     return state_hist_excluding_k.view(len(household_idx), -1)

def get_state_history_excluding_k(household_idx, decision_k, states, time, L):
    """Get state history s_i(t:t-L+1)^{-k} excluding decision type k."""
    if isinstance(household_idx, torch.Tensor):
        household_idx = household_idx.tolist()
    
    start_time = max(0, time - L + 1)
    end_time = min(time + 1, states.shape[1])
    
    state_hist = states[household_idx, start_time:end_time, :]
    other_decisions = [i for i in range(3) if i != decision_k]
    state_hist_excluding_k = state_hist[:, :, other_decisions]
    
    # Calculate expected length
    expected_length = L * 2  # L timesteps × 2 decision types (excluding k)
    actual_length = state_hist_excluding_k.shape[1] * state_hist_excluding_k.shape[2]
    
    # Reshape to flat
    state_hist_flat = state_hist_excluding_k.view(len(household_idx), -1)
    
    # Pad with zeros if necessary (for early timesteps)
    if actual_length < expected_length:
        padding_length = expected_length - actual_length
        padding = torch.zeros(len(household_idx), padding_length, dtype=state_hist.dtype, device=state_hist.device)
        state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
    
    return state_hist_flat


# def get_full_state_history(household_idx, states, time, L):
#     """Get full state history S_i(t:t-L+1) including all decision types."""
#     if isinstance(household_idx, torch.Tensor):
#         household_idx = household_idx.tolist()
    
#     start_time = max(0, time - L + 1)
#     end_time = min(time + 1, states.shape[1])
    
#     state_hist = states[household_idx, start_time:end_time, :]
#     return state_hist.view(len(household_idx), -1)

def get_full_state_history(household_idx, states, time, L):
    """Get full state history S_i(t:t-L+1) including all decision types."""
    if isinstance(household_idx, torch.Tensor):
        household_idx = household_idx.tolist()
    
    start_time = max(0, time - L + 1)
    end_time = min(time + 1, states.shape[1])
    
    state_hist = states[household_idx, start_time:end_time, :]
    
    # Calculate expected length
    expected_length = L * 3  # L timesteps × 3 decision types
    actual_length = state_hist.shape[1] * state_hist.shape[2]  # timesteps × decisions
    # print(f"L: {L}, Actual state history shape: {state_hist.shape}")
    # print(f"Actual state history length: {actual_length}, Expected length: {expected_length}")
    
    # Reshape to flat
    state_hist_flat = state_hist.view(len(household_idx), -1)
    
    # Pad with zeros if necessary (for early timesteps)
    if actual_length < expected_length:
        padding_length = expected_length - actual_length
        padding = torch.zeros(len(household_idx), padding_length, dtype=state_hist.dtype, device=state_hist.device)
        state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
    #print(f"Full state history shape: {state_hist_flat.shape}")
    
    return state_hist_flat


def compute_pairwise_features(features_i, features_j):
    """
    Compute pairwise features f_ij as absolute difference.
    f_ij = |features_i - features_j|
    """
    return torch.abs(features_i - features_j)