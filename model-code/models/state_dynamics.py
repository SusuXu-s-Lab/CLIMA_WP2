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
        """FR-SIC process with f_ij = |features_i - features_j|"""
        batch_size = len(household_idx)
        n_households = features.shape[0]
        
        # Self-activation
        hh_features = features[household_idx]
        state_history_excluding_k = get_state_history_excluding_k(
            household_idx, decision_type, states, time, self.L
        )
        
        decision_onehot = F.one_hot(torch.tensor(decision_type), num_classes=3).float()
        decision_onehot = decision_onehot.unsqueeze(0).expand(batch_size, -1)
        time_tensor = torch.full((batch_size, 1), time, dtype=torch.float32)
        
        p_self = self.self_nn(hh_features, state_history_excluding_k, decision_onehot, time_tensor).squeeze(1)
        
        # Social influence
        influence_probs = []
        
        for batch_idx, hh_i in enumerate(household_idx):
            hh_i_val = hh_i.item()
            
            # Find active neighbors
            active_neighbors = [j for j in range(n_households) 
                              if j != hh_i_val and states[j, time, decision_type] == 1]
            
            if len(active_neighbors) == 0:
                influence_probs.append(torch.tensor(1.0))
                continue
            
            neighbor_influence_probs = []
            
            for j in active_neighbors:
                link_repr = self._get_link_representation(hh_i_val, j, network_data, gumbel_samples, time)
                
                j_state_hist = get_full_state_history([j], states, time, self.L)
                # j_state_hist = get_state_history_excluding_k([j], decision_type, states, time, self.L)
                i_state_hist = get_state_history_excluding_k([hh_i_val], decision_type, states, time, self.L)
                
                feat_i = features[hh_i_val].unsqueeze(0)
                feat_j = features[j].unsqueeze(0)
                dist = distances[hh_i_val, j].unsqueeze(0).unsqueeze(1)
                
                decision_onehot_single = F.one_hot(torch.tensor(decision_type), num_classes=3).float().unsqueeze(0)
                time_tensor_single = torch.full((1, 1), time, dtype=torch.float32)
                
                p_inf = self.influence_nn(link_repr, j_state_hist, i_state_hist, 
                                        feat_i, feat_j, dist, decision_onehot_single, time_tensor_single)
                neighbor_influence_probs.append(p_inf.squeeze())
            
            neighbor_probs = torch.stack(neighbor_influence_probs)
            product_term = torch.prod(1 - neighbor_probs)
            influence_probs.append(product_term)
        
        influence_probs = torch.stack(influence_probs)
        
        # FR-SIC formula
        activation_probs = 1 - (1 - p_self) * influence_probs
        return activation_probs
    
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