"""
blocking_modes.py - Defines different network edge blocking/missing modes
"""
import numpy as np
import pandas as pd


def block_links_temporal(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """
    Temporal random blocking: Independently block p% of edges at each time step.
    The same edge may be present at some time points and missing at others.
    
    Args:
        df: DataFrame with ['household_id_1', 'household_id_2', 'time_step', 'link_type']
        p: Blocking proportion (0-1)
        
    Returns:
        Blocked DataFrame
    """
    df = df.copy()
    to_block_indices = []

    for t in df['time_step'].unique():
        df_t = df[df['time_step'] == t]
        n_block = int(len(df_t) * p)
        if n_block > 0:
            blocked_indices = np.random.choice(df_t.index, size=n_block, replace=False)
            to_block_indices.extend(blocked_indices)

    df_blocked = df.drop(index=to_block_indices)
    return df_blocked


def block_links_structural(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """
    Structural permanent blocking: Randomly select p% of edges (based on household pairs),
    and remove these edges across ALL time steps.
    
    Args:
        df: DataFrame with ['household_id_1', 'household_id_2', 'time_step', 'link_type']
        p: Blocking proportion (0-1)
        
    Returns:
        Blocked DataFrame
    """
    df = df.copy()
    
    # Get unique edge pairs (regardless of time and link_type)
    # Ensure edge direction consistency (smaller ID first)
    df['edge_id_1'] = df[['household_id_1', 'household_id_2']].min(axis=1)
    df['edge_id_2'] = df[['household_id_1', 'household_id_2']].max(axis=1)
    
    # Get all unique edges
    unique_edges = df[['edge_id_1', 'edge_id_2']].drop_duplicates()
    
    # Randomly select edges to permanently block
    n_block = int(len(unique_edges) * p)
    
    if n_block > 0:
        blocked_edges = unique_edges.sample(n=n_block, random_state=None)
        
        # Create set of blocked edges for fast lookup
        blocked_set = set(
            tuple([row['edge_id_1'], row['edge_id_2']])
            for _, row in blocked_edges.iterrows()
        )
        
        # Filter out these edges from all time steps
        mask = df.apply(
            lambda row: tuple([row['edge_id_1'], row['edge_id_2']]) not in blocked_set,
            axis=1
        )
        df_blocked = df[mask].copy()
        
        # Remove temporary columns
        df_blocked = df_blocked.drop(columns=['edge_id_1', 'edge_id_2'])
    else:
        df_blocked = df.drop(columns=['edge_id_1', 'edge_id_2'])
    
    return df_blocked


def apply_blocking_mode(df: pd.DataFrame, mode: str, p: float) -> pd.DataFrame:
    """
    Apply the specified blocking mode to the network dataframe.
    
    Args:
        df: Network DataFrame
        mode: Blocking mode - "temporal" or "structural"
        p: Blocking proportion (0-1)
        
    Returns:
        Blocked DataFrame
    """
    if mode == "temporal":
        return block_links_temporal(df, p)
    elif mode == "structural":
        return block_links_structural(df, p)
    else:
        raise ValueError(f"Unknown blocking mode: {mode}. Use 'temporal' or 'structural'.")