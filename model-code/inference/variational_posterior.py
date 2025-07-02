import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from models import NetworkTypeNN
from models.utils import get_full_state_history

class MeanFieldPosterior:
    """
    Updated variational posterior following PDF formulation.
    
    Key changes:
    1. Compute both conditional probabilities π_ij(t | k) AND marginal probabilities π̄_ij(t)
    2. Call NetworkTypeNN 3 times per (i,j,t) to get full 3x3 transition matrix
    3. Recursive marginal computation: π̄_ij(t) = Σ π̄_ij(t-1)[k'] × π_ij(t | k')[k]
    """
    
    def __init__(self, network_type_nn: NetworkTypeNN, L: int = 1):
        self.network_type_nn = network_type_nn
        self.L = L
        
    def compute_probabilities_batch(self, 
                                   features: torch.Tensor,
                                   states: torch.Tensor,
                                   distances: torch.Tensor,
                                   node_batch: torch.Tensor,
                                   network_data,
                                   max_timestep: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute both conditional and marginal probabilities for batch.
        
        Returns:
            conditional_probs: {pair_key: π_ij(t | k) for k=0,1,2} - shape [3, 3]
            marginal_probs: {pair_key: π̄_ij(t)} - shape [3]
        """
        conditional_probs = {}
        marginal_probs = {}
        
        # Get all pairs involving batch nodes
        batch_pairs_by_time = self._get_batch_pairs(node_batch, network_data, max_timestep, features.shape[0])
        
        # Process each timestep sequentially (important for temporal dependencies)
        for t in range(max_timestep + 1):
            pairs_t = batch_pairs_by_time[t]
            
            if len(pairs_t) == 0:
                continue
            
            # Optimized: one single forward pass generates full 3×3 matrix
            conditionals_t = self._compute_conditional_probabilities_timestep(
                pairs_t, features, states, distances, network_data, t
            )
            
            # Store conditional probabilities
            for idx, (i, j) in enumerate(pairs_t):
                pair_key = f"{i}_{j}_{t}"
                conditional_probs[pair_key] = conditionals_t[idx]  # [3, 3] matrix
            
            # Compute marginal probabilities π̄_ij(t)
            marginals_t = self._compute_marginal_probabilities_timestep(
                pairs_t, conditionals_t, marginal_probs, network_data, t
            )
            
            # Store marginal probabilities  
            for idx, (i, j) in enumerate(pairs_t):
                pair_key = f"{i}_{j}_{t}"
                marginal_probs[pair_key] = marginals_t[idx]  # [3] vector
        
        return conditional_probs, marginal_probs
    
    def _compute_conditional_probabilities_timestep(self, 
                                                   pairs_t: List[Tuple[int, int]],
                                                   features: torch.Tensor,
                                                   states: torch.Tensor, 
                                                   distances: torch.Tensor,
                                                   network_data,
                                                   t: int) -> torch.Tensor:
        """
        Compute conditional probabilities π_ij(t | k) for all pairs at timestep t.
        
        We need to call the neural network 3 times (once for each previous type k'=0,1,2)
        to get the full 3×3 transition matrix.
        
        Returns:
            conditionals: [len(pairs_t), 3, 3] - π_ij(t | k') for k'=0,1,2
        """
        if len(pairs_t) == 0:
            return torch.empty(0, 3, 3)
        
        num_pairs = len(pairs_t)
        conditionals = torch.zeros(num_pairs, 3, 3)
        
        # Prepare common batch data (same for all 3 NN calls)
        batch_features_i = []
        batch_features_j = []
        batch_state_hist_i = []
        batch_state_hist_j = []
        batch_distances = []
        
        for i, j in pairs_t:
            # State histories S_i(t:t-L+1), S_j(t:t-L+1)
            state_hist_i = get_full_state_history([i], states, t, self.L)
            state_hist_j = get_full_state_history([j], states, t, self.L)
            
            batch_features_i.append(features[i])
            batch_features_j.append(features[j])
            batch_state_hist_i.append(state_hist_i.squeeze(0))
            batch_state_hist_j.append(state_hist_j.squeeze(0))
            batch_distances.append(distances[i, j])
        
        # Convert common tensors once
        features_i_batch = torch.stack(batch_features_i)  # [num_pairs, feature_dim]
        features_j_batch = torch.stack(batch_features_j)
        state_hist_i_batch = torch.stack(batch_state_hist_i)      # [num_pairs, L*3]
        state_hist_j_batch = torch.stack(batch_state_hist_j)
        distances_batch = torch.stack(batch_distances).unsqueeze(1)  # [num_pairs, 1]
        is_initial_batch = torch.full((num_pairs, 1), 1.0 if t == 0 else 0.0)  # [num_pairs, 1]
        
        # Single network forward pass
        logits_batch = self.network_type_nn(
            features_i_batch, features_j_batch, state_hist_i_batch, state_hist_j_batch,
            distances_batch, is_initial_batch
        )  # [num_pairs, 3, 3]

        conditionals = F.softmax(logits_batch, dim=2)  # softmax over last dim (k)

        # At t=0 we replicate row 0 for all previous types to keep shape
        if t == 0:
            replicated = conditionals[:, 0, :].unsqueeze(1).repeat(1, 3, 1)
            return replicated

        return conditionals  # [num_pairs, 3, 3]
    
    def _compute_marginal_probabilities_timestep(self,
                                                pairs_t: List[Tuple[int, int]],
                                                conditionals_t: torch.Tensor,
                                                marginal_probs: Dict[str, torch.Tensor],
                                                network_data,
                                                t: int) -> torch.Tensor:
        """
        Compute marginal probabilities π̄_ij(t) using recursive formula.
        
        Base case (t=0): π̄_ij(0) = π_ij(0) (from conditional with dummy previous type)
        Recursive case: π̄_ij(t) = Σ π̄_ij(t-1)[k'] × π_ij(t | k')[k]
        """
        marginals_t = []
        
        for idx, (i, j) in enumerate(pairs_t):
            if t == 0:
                # Base case: use conditional probabilities with dummy previous type (k'=0)
                marginal = conditionals_t[idx, 0, :]  # π_ij(0 | dummy)
            else:
                # Recursive case: get previous marginal
                pair_key_prev = f"{i}_{j}_{t-1}"
                
                if network_data.is_observed(i, j, t-1):
                    # Previous state observed: use one-hot
                    prev_type = network_data.get_link_type(i, j, t-1)
                    π_prev = F.one_hot(torch.tensor(prev_type), num_classes=3).float()
                elif pair_key_prev in marginal_probs:
                    # Previous state hidden: use previous marginal
                    π_prev = marginal_probs[pair_key_prev]
                else:
                    # Fallback: assume no connection
                    π_prev = torch.tensor([1.0, 0.0, 0.0])
                
                # Compute marginal: π̄_ij(t)[k] = Σ_{k'} π̄_ij(t-1)[k'] × π_ij(t | k')[k]
                π_conditional = conditionals_t[idx]  # [3, 3] - π_ij(t | k') for k'=0,1,2
                marginal = torch.sum(π_prev.unsqueeze(1) * π_conditional, dim=0)  # [3]
            
            marginals_t.append(marginal)
        
        return torch.stack(marginals_t) if marginals_t else torch.empty(0, 3)
    
    def _get_batch_pairs(self, node_batch: torch.Tensor, network_data, max_timestep: int, 
                        n_households: int) -> Dict[int, List[Tuple[int, int]]]:
        """Get all pairs involving batch nodes for each timestep."""
        batch_pairs_by_time = {t: [] for t in range(max_timestep + 1)}
        batch_nodes_set = set(node_batch.tolist())
        
        for t in range(max_timestep + 1):
            all_hidden_pairs = network_data.get_hidden_pairs(t)
            for i, j in all_hidden_pairs:
                if i in batch_nodes_set or j in batch_nodes_set:
                    batch_pairs_by_time[t].append((i, j))
        
        return batch_pairs_by_time
    
    # Legacy method for backward compatibility
    def compute_categorical_probabilities_batch(self, *args, **kwargs):
        """Backward compatibility - returns marginal probabilities."""
        conditional_probs, marginal_probs = self.compute_probabilities_batch(*args, **kwargs)
        return marginal_probs