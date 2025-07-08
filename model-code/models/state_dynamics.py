import torch
import torch.nn.functional as F
from typing import Dict

from .neural_networks import InfluenceNN, SelfActivationNN
from .utils import get_state_history_excluding_k,get_full_state_history

class StateTransition:
    """State transition with FR-SIC process using f_ij = |features_i - features_j|"""
    
    def __init__(self, self_nn: SelfActivationNN, influence_nn: InfluenceNN, L: int = 1):
        self.self_nn = self_nn
        self.influence_nn = influence_nn
        self.L = L
        
    def compute_activation_probability(self, household_idx: torch.Tensor, decision_type: int,  # 0=vacant, 1=repair, 2=sell
                                     features: torch.Tensor, states: torch.Tensor, distances: torch.Tensor,
                                     network_data,  # NetworkData object for observed links
                                     gumbel_samples: Dict[str, torch.Tensor], time: int):  # Continuous Gumbel-Softmax samples, time is current timestep
        """
        OPTIMIZED: FR-SIC process with vectorized operations for maximum performance.
        Redirects to the highly optimized vectorized implementation.
        """
        return self.compute_activation_probability_vectorized(
            household_idx, decision_type, features, states, distances, 
            network_data, gumbel_samples, time
        )
    
    def compute_activation_probability_vectorized(self, household_idx: torch.Tensor, decision_type: int,
                                                features: torch.Tensor, states: torch.Tensor, distances: torch.Tensor,
                                                network_data, gumbel_samples: Dict[str, torch.Tensor], time: int):
        """
        HIGHLY OPTIMIZED: Vectorized FR-SIC process computation.
        
        Key optimizations:
        1. Batch processing of state histories
        2. Pre-computation of neighbor relationships
        3. Vectorized neural network calls
        4. Elimination of Python loops where possible
        """
        batch_size = len(household_idx)
        n_households = features.shape[0]
        
        if batch_size == 0:
            return torch.tensor([])
        
        # === OPTIMIZATION 1: Vectorized Self-Activation ===
        hh_features = features[household_idx]  # [batch_size, feature_dim]
        
        # Batch compute state histories for all households at once
        state_history_excluding_k = self._batch_get_state_history_excluding_k(
            household_idx, decision_type, states, time, self.L
        )
        
        # Pre-compute common tensors (avoid recreation)
        decision_onehot = F.one_hot(torch.tensor(decision_type), num_classes=3).float()
        decision_onehot_batch = decision_onehot.unsqueeze(0).expand(batch_size, -1)
        time_tensor_batch = torch.full((batch_size, 1), time, dtype=torch.float32)
        
        # Single NN call for all self-activations
        p_self = self.self_nn(hh_features, state_history_excluding_k, 
                             decision_onehot_batch, time_tensor_batch).squeeze(1)
        
        # === OPTIMIZATION 2: Vectorized Neighbor Influence ===
        # Pre-compute all active neighbors for the batch
        neighbor_data = self._precompute_neighbor_relationships(
            household_idx, states, time, decision_type, n_households
        )
        
        if len(neighbor_data['all_pairs']) == 0:
            # No neighbors for any household
            return p_self
        
        # Batch compute all neighbor influences at once
        influence_probs = self._compute_batch_neighbor_influence(
            neighbor_data, features, states, distances, network_data,
            gumbel_samples, time, decision_type
        )
        
        # === OPTIMIZATION 3: Vectorized Product Computation ===
        final_influence_probs = self._aggregate_neighbor_influences(
            influence_probs, neighbor_data, batch_size
        )
        
        # FR-SIC formula
        activation_probs = 1 - (1 - p_self) * final_influence_probs
        return activation_probs
    
    def _batch_get_state_history_excluding_k(self, household_idx: torch.Tensor, decision_type: int,
                                            states: torch.Tensor, time: int, L: int):
        """
        OPTIMIZED: Batch computation of state histories to avoid repeated calls.
        """
        batch_size = len(household_idx)
        start_time = max(0, time - L + 1)
        end_time = min(time + 1, states.shape[1])
        
        # Batch extract state histories
        state_hist = states[household_idx, start_time:end_time, :]  # [batch_size, time_steps, 3]
        
        # Remove decision_type dimension
        other_decisions = [i for i in range(3) if i != decision_type]
        state_hist_excluding_k = state_hist[:, :, other_decisions]  # [batch_size, time_steps, 2]
        
        # Calculate expected length and pad if necessary
        expected_length = L * 2  # L timesteps × 2 decision types
        actual_timesteps = state_hist_excluding_k.shape[1]
        actual_length = actual_timesteps * 2
        
        # Flatten
        state_hist_flat = state_hist_excluding_k.view(batch_size, -1)
        
        # Pad if necessary
        if actual_length < expected_length:
            padding_length = expected_length - actual_length
            padding = torch.zeros(batch_size, padding_length, 
                                dtype=state_hist.dtype, device=state_hist.device)
            state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
        
        return state_hist_flat
    
    def _precompute_neighbor_relationships(self, household_idx: torch.Tensor, states: torch.Tensor, 
                                         time: int, decision_type: int, n_households: int):
        """
        OPTIMIZED: Pre-compute all neighbor relationships to avoid repeated searches.
        
        Returns:
            dict with 'all_pairs', 'household_to_pairs_map', 'pair_to_households'
        """
        # Find all active neighbors for each household in the batch
        active_state_mask = states[:, time, decision_type] == 1  # [n_households]
        
        all_pairs = []
        household_to_pairs_map = {}  # household_idx -> list of pair indices
        pair_to_households = []  # list of (household_batch_idx, neighbor_idx)
        
        for batch_idx, hh_i in enumerate(household_idx):
            hh_i_val = hh_i.item()
            household_to_pairs_map[batch_idx] = []
            
            # Find active neighbors for this household
            for j in range(n_households):
                if j != hh_i_val and active_state_mask[j]:
                    pair_idx = len(all_pairs)
                    all_pairs.append((hh_i_val, j))
                    household_to_pairs_map[batch_idx].append(pair_idx)
                    pair_to_households.append((batch_idx, j))
        
        return {
            'all_pairs': all_pairs,
            'household_to_pairs_map': household_to_pairs_map,
            'pair_to_households': pair_to_households
        }
    
    def _compute_batch_neighbor_influence(self, neighbor_data: dict, features: torch.Tensor,
                                        states: torch.Tensor, distances: torch.Tensor,
                                        network_data, gumbel_samples: Dict[str, torch.Tensor],
                                        time: int, decision_type: int):
        """
        OPTIMIZED: Compute all neighbor influences in a single batch call.
        """
        all_pairs = neighbor_data['all_pairs']
        pair_to_households = neighbor_data['pair_to_households']
        
        if len(all_pairs) == 0:
            return torch.tensor([])
        
        n_pairs = len(all_pairs)
        
        # Pre-allocate tensors for batch processing
        batch_link_reprs = []
        batch_j_state_hists = []
        batch_i_state_hists = []
        batch_feat_i = []
        batch_feat_j = []
        batch_distances = []
        
        # Batch collect all neighbor state histories
        j_indices = [j for _, j in pair_to_households]
        if len(j_indices) > 0:
            j_indices_tensor = torch.tensor(j_indices, dtype=torch.long)
            batch_j_state_hists_tensor = self._batch_get_full_state_history(
                j_indices_tensor, states, time, self.L
            )
        
        # Process each pair efficiently
        for pair_idx, ((hh_i, j), (batch_idx, _)) in enumerate(zip(all_pairs, pair_to_households)):
            # Link representation
            link_repr = self._get_link_representation(hh_i, j, network_data, gumbel_samples, time)
            batch_link_reprs.append(link_repr.squeeze(0))
            
            # State histories (j already computed in batch)
            batch_j_state_hists.append(batch_j_state_hists_tensor[pair_idx])
            
            # i state history (compute individually for now, could be optimized further)
            i_state_hist = get_state_history_excluding_k([hh_i], decision_type, states, time, self.L)
            batch_i_state_hists.append(i_state_hist.squeeze(0))
            
            # Features and distances
            batch_feat_i.append(features[hh_i])
            batch_feat_j.append(features[j])
            batch_distances.append(distances[hh_i, j].unsqueeze(0))
        
        if len(batch_link_reprs) == 0:
            return torch.tensor([])
        
        # Convert to tensors
        link_reprs_tensor = torch.stack(batch_link_reprs)  # [n_pairs, 3]
        j_state_hists_tensor = torch.stack(batch_j_state_hists)  # [n_pairs, L*3]
        i_state_hists_tensor = torch.stack(batch_i_state_hists)  # [n_pairs, L*2]
        feat_i_tensor = torch.stack(batch_feat_i)  # [n_pairs, feature_dim]
        feat_j_tensor = torch.stack(batch_feat_j)  # [n_pairs, feature_dim]
        distances_tensor = torch.stack(batch_distances)  # [n_pairs, 1]
        
        # Common tensors for the batch
        decision_onehot_tensor = F.one_hot(torch.tensor(decision_type), num_classes=3).float()
        decision_onehot_batch = decision_onehot_tensor.unsqueeze(0).expand(n_pairs, -1)
        time_tensor_batch = torch.full((n_pairs, 1), time, dtype=torch.float32)
        
        # Single batch call to influence neural network
        influence_probs = self.influence_nn(
            link_reprs_tensor, j_state_hists_tensor, i_state_hists_tensor,
            feat_i_tensor, feat_j_tensor, distances_tensor,
            decision_onehot_batch, time_tensor_batch
        ).squeeze(1)
        
        return influence_probs
    
    def _batch_get_full_state_history(self, household_idx: torch.Tensor, states: torch.Tensor, 
                                     time: int, L: int):
        """
        OPTIMIZED: Batch version of get_full_state_history.
        """
        batch_size = len(household_idx)
        start_time = max(0, time - L + 1)
        end_time = min(time + 1, states.shape[1])
        
        # Batch extract state histories
        state_hist = states[household_idx, start_time:end_time, :]  # [batch_size, time_steps, 3]
        
        # Calculate expected length
        expected_length = L * 3  # L timesteps × 3 decision types
        actual_timesteps = state_hist.shape[1]
        actual_length = actual_timesteps * 3
        
        # Flatten
        state_hist_flat = state_hist.view(batch_size, -1)
        
        # Pad if necessary
        if actual_length < expected_length:
            padding_length = expected_length - actual_length
            padding = torch.zeros(batch_size, padding_length, 
                                dtype=state_hist.dtype, device=state_hist.device)
            state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
        
        return state_hist_flat
    
    def _aggregate_neighbor_influences(self, influence_probs: torch.Tensor, neighbor_data: dict, 
                                     batch_size: int):
        """
        OPTIMIZED: Aggregate neighbor influences using vectorized operations where possible.
        """
        household_to_pairs_map = neighbor_data['household_to_pairs_map']
        
        final_influence_probs = torch.ones(batch_size, dtype=torch.float32)
        
        for batch_idx in range(batch_size):
            pair_indices = household_to_pairs_map[batch_idx]
            
            if len(pair_indices) == 0:
                # No neighbors, influence probability remains 1.0
                continue
            
            # Get influence probabilities for this household's neighbors
            neighbor_probs = influence_probs[pair_indices]
            
            # Compute product term: ∏(1 - p_inf)
            product_term = torch.prod(1 - neighbor_probs)
            final_influence_probs[batch_idx] = product_term
        
        return final_influence_probs
        
    def _get_link_representation(self, i, j, network_data, gumbel_samples, time):
        """Get continuous link representation."""
        i, j = min(i, j), max(i, j)
        
        if network_data.is_observed(i, j, time) & time<=15:
            link_type = network_data.get_link_type(i, j, time)
            link_repr = F.one_hot(torch.tensor(link_type), num_classes=3).float()
        else:
            pair_key = f"{i}_{j}_{time}"
            link_repr = gumbel_samples.get(pair_key, torch.tensor([1.0, 0.0, 0.0]))
        
        return link_repr.unsqueeze(0)
    

    def compute_detailed_activation_probability(self, household_idx: torch.Tensor, decision_type: int,
                                        features: torch.Tensor, states: torch.Tensor, distances: torch.Tensor,
                                        network_data, gumbel_samples: Dict[str, torch.Tensor], time: int):
        """
        Detailed version that returns self-activation and neighbor-by-neighbor influence probabilities.
        Only used for evaluation, not training.
        """
        batch_size = len(household_idx)
        n_households = features.shape[0]
        
        detailed_results = []
        
        for batch_idx, hh_i in enumerate(household_idx):
            hh_i_val = hh_i.item()
            
            # 1. SELF-ACTIVATION PROBABILITY
            hh_features = features[hh_i_val].unsqueeze(0)
            state_history_excluding_k = get_state_history_excluding_k(
                [hh_i_val], decision_type, states, time, self.L
            )
            
            decision_onehot = F.one_hot(torch.tensor(decision_type), num_classes=3).float().unsqueeze(0)
            time_tensor = torch.full((1, 1), time, dtype=torch.float32)
            
            p_self = self.self_nn(hh_features, state_history_excluding_k, decision_onehot, time_tensor).squeeze().item()
            
            # 2. NEIGHBOR-BY-NEIGHBOR INFLUENCE PROBABILITIES
            active_neighbors = []
            neighbor_influences = []

            for j in range(n_households):
                if j == hh_i_val:
                    continue
                    
                # First check if there's a connection
                link_repr = self._get_link_representation(hh_i_val, j, network_data, gumbel_samples, time)
                link_type = torch.argmax(link_repr).item()
                
                if link_type > 0:  # Has connection
                    # Then check if neighbor is active
                    if states[j, time, decision_type] == 1:  # Neighbor is active
                        # Calculate influence probability
                        active_neighbors.append(j)
                        j_state_hist = get_full_state_history([j], states, time, self.L)
                        i_state_hist = get_state_history_excluding_k([hh_i_val], decision_type, states, time, self.L)
                        
                        feat_i = features[hh_i_val].unsqueeze(0)
                        feat_j = features[j].unsqueeze(0)
                        dist = distances[hh_i_val, j].unsqueeze(0).unsqueeze(1)
                        
                        p_inf = self.influence_nn(link_repr, j_state_hist, i_state_hist, 
                                                feat_i, feat_j, dist, decision_onehot, time_tensor).squeeze().item()
                        
                        neighbor_influences.append({
                            'neighbor_id': j,
                            'link_type': link_type,
                            'link_probs': link_repr.squeeze().tolist(),
                            'influence_prob': p_inf,
                            'distance': distances[hh_i_val, j].item()
                        })

            
            # 3. COMPUTE FINAL PROBABILITY STEP BY STEP
            if len(neighbor_influences) == 0:
                final_activation_prob = p_self
                social_influence_term = 0.0
            else:
                neighbor_probs = [ni['influence_prob'] for ni in neighbor_influences]
                product_term = torch.prod(torch.tensor([1 - p for p in neighbor_probs])).item()
                social_influence_term = 1 - product_term
                final_activation_prob = 1 - (1 - p_self) * product_term
            
            detailed_results.append({
                'household_id': hh_i_val,
                'decision_type': decision_type,
                'timestep': time,
                'self_activation_prob': p_self,
                'active_neighbors': len(active_neighbors),
                'neighbor_influences': neighbor_influences,
                'social_influence_term': social_influence_term,
                'final_activation_prob': final_activation_prob
            })
        
        # Return both detailed results and the original activation probabilities
        final_probs = torch.tensor([r['final_activation_prob'] for r in detailed_results])
        return final_probs, detailed_results