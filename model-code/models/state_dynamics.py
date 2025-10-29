import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .neural_networks import SeqSelfNN, SeqPairInfluenceNN
from .utils import compute_pairwise_features


class StateTransition:
    """State transition with seq2seq GRU architecture"""

    def __init__(self, seq_self_nn: SeqSelfNN = None, seq_pair_infl_nn: SeqPairInfluenceNN = None):
        # Initialize GRU-based networks with lazy initialization
        self.seq_self_nn = seq_self_nn if seq_self_nn is not None else SeqSelfNN(feature_dim=None, hidden_dim=32)
        self.seq_pair_infl_nn = seq_pair_infl_nn if seq_pair_infl_nn is not None else SeqPairInfluenceNN(pairwise_feature_dim=None, hidden_dim=32)

    def compute_activation_probability_seq(self,
                                          features: torch.Tensor,
                                          states: torch.Tensor,
                                          distances: torch.Tensor,
                                          network_data,
                                          gumbel_samples_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute activation probabilities for all timesteps using seq2seq GRU.
        
        Args:
            features: [N, feature_dim] - static node features
            states: [N, T+1, 3] - state tensor (including t=0)
            distances: [N, N] - pairwise distances
            network_data: network structure with neighbor_index
            gumbel_samples_dict: Dict[f"{i}_{j}_{t}", Tensor[3]] - gumbel link probabilities
            
        Returns:
            p_next_seq: [N, T, 3] - activation probabilities for t=1..T
        """
        device = states.device
        N, T_plus_1, K = states.shape
        T = T_plus_1 - 1
        
        # ========== 1. Self Channel ==========
        # Input: [s_i(t), x_i] for t=0..T-1
        states_seq = states[:, :-1, :].float()  # [N, T, 3] - exclude last timestep
        features_rep = features.unsqueeze(1).expand(N, T, -1)  # [N, T, feature_dim]
        X_self = torch.cat([states_seq, features_rep], dim=-1)  # [N, T, 3+feature_dim]
        
        h_self_seq, _ = self.seq_self_nn(X_self)  # [N, T, 3]
        
        # Susceptible mask: only predict for inactive positions
        susceptible = 1.0 - states_seq  # [N, T, 3]
        h_self_masked = h_self_seq * susceptible  # [N, T, 3]
        
        # ========== 2. Neighbor Channel ==========
        # Build pair sequences using precomputed neighbor relationships
        all_pairs, pair_to_i_map = self._build_pair_sequences(
            N, T, states, features, distances, network_data, gumbel_samples_dict
        )
        
        if len(all_pairs) == 0:
            # No pairs - only self-activation
            p_next_seq = h_self_masked.clamp(1e-6, 1-1e-6)
            return p_next_seq
        
        # Stack all pair sequences: [P, T, input_dim]
        X_pair = torch.stack(all_pairs, dim=0)  # [P, T, 10+pairwise_feat_dim]
        
        # Forward through pair GRU
        h_pair_seq, _ = self.seq_pair_infl_nn(X_pair)  # [P, T, 3]
        
        # Apply triple-gate and aggregate via noisy-OR
        h_neigh_seq = self._aggregate_neighbor_influence(
            h_pair_seq, pair_to_i_map, states_seq, X_pair, N, T
        )  # [N, T, 3]
        
        # ========== 3. Final Combination ==========
        # p(t+1) = 1 - (1 - h_self) * (1 - h_neigh)
        p_next_seq = 1.0 - (1.0 - h_self_masked) * (1.0 - h_neigh_seq)
        p_next_seq = p_next_seq.clamp(1e-6, 1-1e-6)
        
        return p_next_seq
    
    def compute_activation_probability_seq_detailed(self,
                                                   features: torch.Tensor,
                                                   states: torch.Tensor,
                                                   distances: torch.Tensor,
                                                   network_data,
                                                   gumbel_samples_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Same as compute_activation_probability_seq but returns detailed breakdown.
        
        Returns:
            p_next_seq: [N, T, 3] - final activation probabilities
            detailed_results: List of dicts with per-household, per-timestep breakdown
        """
        device = states.device
        N, T_plus_1, K = states.shape
        T = T_plus_1 - 1
        
        # ========== 1. Self Channel ==========
        states_seq = states[:, :-1, :].float()
        features_rep = features.unsqueeze(1).expand(N, T, -1)
        X_self = torch.cat([states_seq, features_rep], dim=-1)
        
        h_self_seq, _ = self.seq_self_nn(X_self)  # [N, T, 3]
        susceptible = 1.0 - states_seq
        h_self_masked = h_self_seq * susceptible
        
        # ========== 2. Neighbor Channel ==========
        all_pairs, pair_to_i_map, pair_metadata = self._build_pair_sequences_detailed(
            N, T, states, features, distances, network_data, gumbel_samples_dict
        )
        
        if len(all_pairs) == 0:
            p_next_seq = h_self_masked.clamp(1e-6, 1-1e-6)
            detailed_results = self._format_detailed_results_no_neighbors(
                N, T, h_self_masked, susceptible
            )
            return p_next_seq, detailed_results
        
        X_pair = torch.stack(all_pairs, dim=0)
        h_pair_seq, _ = self.seq_pair_infl_nn(X_pair)  # [P, T, 3]
        
        h_neigh_seq, neighbor_details = self._aggregate_neighbor_influence_detailed(
            h_pair_seq, pair_to_i_map, states_seq, X_pair, N, T, pair_metadata
        )
        
        # ========== 3. Final Combination ==========
        p_next_seq = 1.0 - (1.0 - h_self_masked) * (1.0 - h_neigh_seq)
        p_next_seq = p_next_seq.clamp(1e-6, 1-1e-6)
        
        # ========== 4. Format Detailed Results ==========
        detailed_results = []
        for i in range(N):
            for t in range(T):
                result = {
                    'household_id': int(i),
                    'timestep': int(t + 1),  # Predicting for t+1
                    'self_activation_prob': h_self_masked[i, t].detach().cpu().numpy(),
                    'self_mask': susceptible[i, t].detach().cpu().numpy(),
                    'neighbor_influences': neighbor_details.get((i, t), []),
                    'neighbor_noisy_or': h_neigh_seq[i, t].detach().cpu().numpy(),
                    'final_activation_prob': p_next_seq[i, t].detach().cpu().numpy()
                }
                detailed_results.append(result)
        
        return p_next_seq, detailed_results
    
    def _build_pair_sequences(self, N: int, T: int, states: torch.Tensor, features: torch.Tensor,
                             distances: torch.Tensor, network_data, gumbel_samples_dict: Dict) -> Tuple[List, Dict]:
        """
        Build pair sequence inputs [P, T, input_dim] using neighbor filtering.
        
        Returns:
            all_pairs: List of tensors [T, input_dim] for each pair
            pair_to_i_map: Dict mapping pair_idx -> target_household_i
        """
        device = states.device
        neighbor_index = getattr(network_data, "neighbor_index", None)
        
        # Reorganize gumbel samples into [i][j][t] structure
        gumbel_tensor = self._reorganize_gumbel_samples(gumbel_samples_dict, N, T, device)
        
        all_pairs = []
        pair_to_i_map = {}
        
        # Iterate through all candidate pairs
        for i in range(N):
            candidates = range(N) if neighbor_index is None else neighbor_index[i]
            
            for j in candidates:
                if j == i:
                    continue
                
                # Build sequence for this pair: [T, input_dim]
                pair_seq = self._build_single_pair_sequence(
                    i, j, T, states, features, distances, gumbel_tensor
                )
                
                pair_idx = len(all_pairs)
                all_pairs.append(pair_seq)
                pair_to_i_map[pair_idx] = i
        
        return all_pairs, pair_to_i_map
    
    def _build_pair_sequences_detailed(self, N: int, T: int, states: torch.Tensor, features: torch.Tensor,
                                      distances: torch.Tensor, network_data, gumbel_samples_dict: Dict):
        """Same as _build_pair_sequences but also returns pair metadata for detailed output"""
        device = states.device
        neighbor_index = getattr(network_data, "neighbor_index", None)
        
        gumbel_tensor = self._reorganize_gumbel_samples(gumbel_samples_dict, N, T, device)
        
        all_pairs = []
        pair_to_i_map = {}
        pair_metadata = {}  # pair_idx -> (i, j)
        
        for i in range(N):
            candidates = range(N) if neighbor_index is None else neighbor_index[i]
            
            for j in candidates:
                if j == i:
                    continue
                
                pair_seq = self._build_single_pair_sequence(
                    i, j, T, states, features, distances, gumbel_tensor
                )
                
                pair_idx = len(all_pairs)
                all_pairs.append(pair_seq)
                pair_to_i_map[pair_idx] = i
                pair_metadata[pair_idx] = (i, j)
        
        return all_pairs, pair_to_i_map, pair_metadata
    
    def _build_single_pair_sequence(self, i: int, j: int, T: int, states: torch.Tensor,
                                   features: torch.Tensor, distances: torch.Tensor,
                                   gumbel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Build input sequence for a single pair (i, j).
        
        Returns:
            pair_seq: [T, 10+pairwise_feat_dim] - input sequence
        """
        # Get link probabilities for all timesteps: [T, 3]
        i_min, j_max = min(i, j), max(i, j)
        g_seq = gumbel_tensor[i_min, j_max, :, :]  # [T, 3]
        
        # State sequences: [T, 3]
        s_i_seq = states[i, :-1, :].float()  # Exclude last timestep
        s_j_seq = states[j, :-1, :].float()
        
        # Static pairwise features
        f_ij = compute_pairwise_features(
            features[i].unsqueeze(0), features[j].unsqueeze(0)
        ).squeeze(0)  # [pairwise_feat_dim]
        d_ij = distances[i, j].unsqueeze(0)  # [1]
        
        static_features = torch.cat([f_ij, d_ij], dim=0)  # [pairwise_feat_dim+1]
        static_rep = static_features.unsqueeze(0).expand(T, -1)  # [T, pairwise_feat_dim+1]
        
        # Concatenate: [g_ij(t), s_i(t), s_j(t), static]
        pair_seq = torch.cat([g_seq, s_i_seq, s_j_seq, static_rep], dim=-1)  # [T, 10+pairwise_feat_dim]
        
        return pair_seq
    
    def _reorganize_gumbel_samples(self, gumbel_samples_dict: Dict, N: int, T: int, device) -> torch.Tensor:
        """
        Reorganize gumbel samples from dict to tensor.
        
        Args:
            gumbel_samples_dict: Dict[f"{i}_{j}_{t}", Tensor[3]]
            
        Returns:
            gumbel_tensor: [N, N, T, 3] - link probabilities
        """
        gumbel_tensor = torch.zeros((N, N, T, 3), device=device)
        gumbel_tensor[:, :, :, 0] = 1.0  # Default to no-link
        
        for pair_key, link_probs in gumbel_samples_dict.items():
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            
            i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Ensure i < j for consistent indexing
            i_min, j_max = min(i, j), max(i, j)
            
            if t < T:
                gumbel_tensor[i_min, j_max, t, :] = link_probs.to(device)
        
        return gumbel_tensor
    
    def _aggregate_neighbor_influence(self, h_pair_seq: torch.Tensor, pair_to_i_map: Dict,
                                     states_seq: torch.Tensor, X_pair: torch.Tensor,
                                     N: int, T: int) -> torch.Tensor:
        """
        Apply triple-gate and aggregate neighbor influences via noisy-OR.
        
        Args:
            h_pair_seq: [P, T, 3] - raw neighbor hazards
            pair_to_i_map: Dict[pair_idx -> target_i]
            states_seq: [N, T, 3] - state sequence
            X_pair: [P, T, 10+feat_dim] - pair input (for extracting g_ij and s_j)
            
        Returns:
            h_neigh_seq: [N, T, 3] - aggregated neighbor influence
        """
        device = h_pair_seq.device
        P = h_pair_seq.shape[0]
        
        # Extract g_ij and s_j from X_pair
        g_ij_seq = X_pair[:, :, :3]  # [P, T, 3] - link probabilities
        s_j_seq = X_pair[:, :, 6:9]  # [P, T, 3] - source states (after s_i which is at 3:6)
        
        # Triple gate for each pair and timestep
        p_exist = (g_ij_seq[:, :, 1] + g_ij_seq[:, :, 2]).unsqueeze(-1).expand(P, T, 3)  # [P, T, 3]
        
        # For each pair, get target i's susceptibility
        i_indices = torch.tensor([pair_to_i_map[p] for p in range(P)], dtype=torch.long, device=device)
        susceptible_i = 1.0 - states_seq[i_indices, :, :]  # [P, T, 3]
        
        # Apply gates: p_exist * s_j * susceptible_i
        neighbor_mask = p_exist * s_j_seq * susceptible_i  # [P, T, 3]
        h_pair_masked = h_pair_seq * neighbor_mask  # [P, T, 3]
        
        # Aggregate via noisy-OR: h_neigh[i] = 1 - prod_j(1 - h_j)
        # Use log-space for numerical stability
        log_complement_sum = torch.zeros((N, T, 3), device=device)
        
        for p_idx in range(P):
            target_i = pair_to_i_map[p_idx]
            # Accumulate log(1 - h_j)
            log_complement_sum[target_i] += torch.log1p(-h_pair_masked[p_idx].clamp(0, 1-1e-8))
        
        # Convert back: 1 - exp(sum(log(1-h_j))) = 1 - prod(1-h_j)
        h_neigh_seq = 1.0 - torch.exp(log_complement_sum)
        
        return h_neigh_seq.clamp(0, 1-1e-8)
    
    def _aggregate_neighbor_influence_detailed(self, h_pair_seq: torch.Tensor, pair_to_i_map: Dict,
                                              states_seq: torch.Tensor, X_pair: torch.Tensor,
                                              N: int, T: int, pair_metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Same as _aggregate_neighbor_influence but returns detailed per-neighbor info"""
        device = h_pair_seq.device
        P = h_pair_seq.shape[0]
        
        g_ij_seq = X_pair[:, :, :3]
        s_j_seq = X_pair[:, :, 6:9]
        
        p_exist = (g_ij_seq[:, :, 1] + g_ij_seq[:, :, 2]).unsqueeze(-1).expand(P, T, 3)
        i_indices = torch.tensor([pair_to_i_map[p] for p in range(P)], dtype=torch.long, device=device)
        susceptible_i = 1.0 - states_seq[i_indices, :, :]
        
        neighbor_mask = p_exist * s_j_seq * susceptible_i
        h_pair_masked = h_pair_seq * neighbor_mask
        
        # Collect details first
        neighbor_details = {}  # (i, t) -> list of neighbor info
        
        for p_idx in range(P):
            target_i = pair_to_i_map[p_idx]
            source_j = pair_metadata[p_idx][1]
            
            for t in range(T):
                key = (target_i, t)
                if key not in neighbor_details:
                    neighbor_details[key] = []
                
                # Record neighbor influence for each decision type
                for k in range(3):
                    if neighbor_mask[p_idx, t, k] > 0:
                        neighbor_details[key].append({
                            'neighbor_id': int(source_j),
                            'decision_type': int(k),
                            'link_probs': g_ij_seq[p_idx, t].detach().cpu().numpy(),
                            'influence_prob': h_pair_masked[p_idx, t, k].item(),  # Store masked value
                            'mask': neighbor_mask[p_idx, t, k].item()
                        })
        
        # Aggregate via noisy-OR using log-space
        log_complement_sum = torch.zeros((N, T, 3), device=device)
        
        for p_idx in range(P):
            target_i = pair_to_i_map[p_idx]
            log_complement_sum[target_i] += torch.log1p(-h_pair_masked[p_idx].clamp(0, 1-1e-8))
        
        h_neigh_seq = 1.0 - torch.exp(log_complement_sum)
        
        return h_neigh_seq.clamp(0, 1-1e-8), neighbor_details
    
    def _format_detailed_results_no_neighbors(self, N: int, T: int, h_self_masked: torch.Tensor,
                                             susceptible: torch.Tensor) -> List[Dict]:
        """Format detailed results when there are no neighbor pairs"""
        detailed_results = []
        for i in range(N):
            for t in range(T):
                result = {
                    'household_id': int(i),
                    'timestep': int(t + 1),
                    'self_activation_prob': h_self_masked[i, t].detach().cpu().numpy(),
                    'self_mask': susceptible[i, t].detach().cpu().numpy(),
                    'neighbor_influences': [],
                    'neighbor_noisy_or': torch.zeros(3).numpy(),
                    'final_activation_prob': h_self_masked[i, t].detach().cpu().numpy()
                }
                detailed_results.append(result)
        return detailed_results