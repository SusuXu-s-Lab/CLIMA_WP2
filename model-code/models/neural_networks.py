import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_pairwise_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_pairwise_features

class NetworkTypeNN(nn.Module):
    """
    Updated NN_type: Input [f_ij(t), S_i(t:t-L+1), S_j(t:t-L+1), prev_link_type, dist_ij, is_initial]
    
    Changes from original:
    1. Added prev_link_type (3-dim one-hot) 
    2. Added is_initial flag (1-dim)
    3. Restored temporal history parameter L
    4. Will be called 3 times per (i,j,t) to get full 3x3 transition matrix
    """
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.L = L
        # Input: f_ij + S_i(t:t-L+1) + S_j(t:t-L+1) + prev_link_type + dist_ij + is_initial
        # Dimensions: feature_dim + L*3 + L*3 + 3 + 1 + 1
        input_dim = feature_dim + L * 3 + L * 3 + 3 + 1 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output 3 logits (will be softmaxed)
        )
    
    def forward(self, features_i, features_j, state_history_i, state_history_j, prev_link_type, distances, is_initial):
        """
        Args:
            features_i, features_j: [batch_size, feature_dim]
            state_history_i, state_history_j: [batch_size, L*3] - state history S_i(t:t-L+1), S_j(t:t-L+1)
            prev_link_type: [batch_size, 3] - one-hot encoding of previous link type
            distances: [batch_size, 1] 
            is_initial: [batch_size, 1] - 1 if t=0, 0 otherwise
        
        Returns:
            logits: [batch_size, 3] - will be softmaxed to get π_ij(t | prev_type)
        """
        f_ij = compute_pairwise_features(features_i, features_j)
        x = torch.cat([f_ij, state_history_i, state_history_j, prev_link_type, distances, is_initial], dim=1)
        return self.network(x)


class SelfActivationNN(nn.Module):
    """NN_self: Input [x_i(t), s_i(t:t-L+1)^{-k}, k, t] - unchanged"""
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.L = L
        input_dim = feature_dim + L * 2 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features, state_history_excluding_k, decision_type_onehot, time):
        x = torch.cat([features, state_history_excluding_k, decision_type_onehot, time], dim=1)
        return torch.sigmoid(self.network(x))


class InfluenceNN(nn.Module):
    """NN_influence: Input [ℓ_ij(t), s_j^{-k}, s_i^{-k}, f_ij(t), dist_ij, k, t] - unchanged"""
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.L = L
        # input_dim = 3 + L*2 + L*2 + feature_dim + 1 + 3 + 1
        input_dim = 3 + L*2 + L*3 + feature_dim + 1 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    # def forward(self, link_repr, state_history_j_excluding_k, state_history_i_excluding_k,
    #             features_i, features_j, distances, decision_type_onehot, time):
    #     f_ij = compute_pairwise_features(features_i, features_j)
    #     x = torch.cat([link_repr, state_history_j_excluding_k, state_history_i_excluding_k,
    #                    f_ij, distances, decision_type_onehot, time], dim=1)
    #     return torch.sigmoid(self.network(x))

    def forward(self, link_repr, state_history_j_excluding_k, state_history_i_excluding_k,
                features_i, features_j, distances, decision_type_onehot, time):
        
        f_ij = compute_pairwise_features(features_i, features_j)
        x = torch.cat([link_repr, state_history_j_excluding_k, state_history_i_excluding_k,
                       f_ij, distances, decision_type_onehot, time], dim=1)
        
        base_influence = torch.sigmoid(self.network(x))
        
        # link_repr[0] = P(no link), link_repr[1] = P(bonding), link_repr[2] = P(bridging)
        connection_strength = link_repr[:, 1] + link_repr[:, 2]  # 1 - P(no link)

        gated_influence = base_influence.squeeze() * connection_strength
        
        return gated_influence.unsqueeze(-1)


class InteractionFormationNN(nn.Module):
    """NN_form: Input [f_ij(t), s_i(t), s_j(t), dist_ij] - unchanged"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        input_dim = feature_dim + 3 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features_i, features_j, states_i, states_j, distances):
        f_ij = compute_pairwise_features(features_i, features_j)
        x = torch.cat([f_ij, states_i, states_j, distances], dim=1)
        return torch.sigmoid(self.network(x))

# class InteractionFormationNN(nn.Module):
#     """Logistic regression version: Input [f_ij(t), s_i(t), s_j(t), dist_ij]"""
    
#     def __init__(self, feature_dim: int, hidden_dim: int = 32):  # hidden_dim ignored now
#         super().__init__()
        
#         # Instead of full NN, just learnable weights for logistic regression
#         self.demographic_weight = nn.Parameter(torch.tensor(1.0))
#         self.state_alignment_weight = nn.Parameter(torch.tensor(1.0))
#         self.geographic_weight = nn.Parameter(torch.tensor(1.0))
#         self.cross_term_weight = nn.Parameter(torch.tensor(0.0))  # Interaction term
#         self.bias = nn.Parameter(torch.tensor(0.0))
        
#         # Fixed normalization factors (you can tune these)
#         self.register_buffer('demo_scale', torch.tensor(1.0))
#         self.register_buffer('geo_scale', torch.tensor(1.0))
    
#     def forward(self, features_i, features_j, states_i, states_j, distances):
#         f_ij = compute_pairwise_features(features_i, features_j)  # Your existing function
        
#         # Engineer interpretable features
#         # 1. Demographic similarity (based on your f_ij)
#         demographic_similarity = torch.exp(-torch.sum(f_ij**2, dim=1) / self.demo_scale)
        
#         # 2. State alignment (how similar their current states are)
#         state_alignment = torch.sum(states_i * states_j, dim=1)
        
#         # 3. Geographic proximity
#         geographic_proximity = torch.exp(-distances.squeeze()**2 / self.geo_scale)
        
#         # 4. Cross-term (demographic × state interaction)
#         cross_term = demographic_similarity * state_alignment
        
#         # Logistic regression: linear combination + sigmoid
#         linear_combination = (self.demographic_weight * demographic_similarity +
#                             self.state_alignment_weight * state_alignment + 
#                             self.geographic_weight * geographic_proximity +
#                             self.cross_term_weight * cross_term +
#                             self.bias)
        
#         return torch.sigmoid(linear_combination)


class SelfActivationNN(nn.Module):
    """NN_self: Input [x_i(t), s_i(t:t-L+1)^{-k}, k, t] - unchanged"""
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.L = L
        input_dim = feature_dim + L * 2 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features, state_history_excluding_k, decision_type_onehot, time):
        x = torch.cat([features, state_history_excluding_k, decision_type_onehot, time], dim=1)
        return torch.sigmoid(self.network(x))


class InteractionFormationNN(nn.Module):
    """NN_form: Input [f_ij(t), s_i(t), s_j(t), dist_ij] - unchanged"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        input_dim = feature_dim + 3 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features_i, features_j, states_i, states_j, distances):
        f_ij = compute_pairwise_features(features_i, features_j)
        x = torch.cat([f_ij, states_i, states_j, distances], dim=1)
        return torch.sigmoid(self.network(x))