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



class InfluenceNN(nn.Module):
    """
    NN_influence: Input [ℓ_ij(t), s_j^{-k}, s_i^{-k}, f_ij(t), dist_ij, k, t, (optional: RBF_features, inactive_fraction)]
    
    Enhanced features (when use_enhanced_features=True):
    - RBF temporal features: Encode time distribution
    - Inactive fraction: Population saturation info
    """
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64, num_layers: int = 2,
                 use_enhanced_features: bool = False, num_rbf_centers: int = 5,
                 rbf_sigma: float = 3.0, max_time: int = 23):
        super().__init__()
        self.L = L
        self.num_layers = num_layers
        self.use_enhanced_features = use_enhanced_features
        self.num_rbf_centers = num_rbf_centers
        self.rbf_sigma = rbf_sigma
        self.max_time = max_time
        
        # Base input
        base_input_dim = 3 + L*2 + L*3 + feature_dim + 1 + 3 + 1
        
        # Enhanced features (when enabled)
        enhanced_dim = 0
        if use_enhanced_features:
            # Only inactive_fraction now (RBF disabled)
            enhanced_dim = 1  # inactive_fraction only
        
        input_dim = base_input_dim + enhanced_dim
        
        # Register RBF centers
        if use_enhanced_features:
            centers = torch.linspace(0, max_time, num_rbf_centers)
            self.register_buffer('rbf_centers', centers)
        
        # Build flexible multi-layer network
        layers = []
        
        if num_layers == 2:
            # Original 2-layer architecture (backward compatible)
            layers.extend([
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ])
        elif num_layers == 3:
            # 3-layer architecture (medium capacity)
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ])
        elif num_layers == 4:
            # 4-layer architecture (high capacity)
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ])
        else:
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")
        
        self.network = nn.Sequential(*layers)
    
    def _compute_rbf_features(self, time: torch.Tensor) -> torch.Tensor:
        """Compute RBF features for time encoding."""
        batch_size = time.shape[0]
        time_expanded = time.expand(-1, self.num_rbf_centers)
        centers_expanded = self.rbf_centers.unsqueeze(0).expand(batch_size, -1)
        rbf_features = torch.exp(-((time_expanded - centers_expanded) ** 2) / (2 * self.rbf_sigma ** 2))
        return rbf_features

    def forward(self, link_repr, state_history_j_excluding_k, state_history_i_excluding_k,
                features_i, features_j, distances, decision_type_onehot, time, 
                inactive_fraction=None):
        """
        Args:
            inactive_fraction: [batch_size, 1] - Fraction of population still inactive
                              Only used when use_enhanced_features=True
        """
        f_ij = compute_pairwise_features(features_i, features_j)
        
        # Base features
        base_input = [link_repr, state_history_j_excluding_k, state_history_i_excluding_k,
                      f_ij, distances, decision_type_onehot, time]
        
        # ✅ Enhanced features DISABLED (inactive_fraction caused issues)
        # if self.use_enhanced_features:
        #     # 1. RBF temporal encoding - DISABLED (too strong, causes time-only fitting)
        #     # rbf_features = self._compute_rbf_features(time)
        #     # base_input.append(rbf_features)
        #     
        #     # 2. Population saturation info
        #     if inactive_fraction is None:
        #         inactive_fraction = torch.full((features_i.shape[0], 1), 0.5, device=features_i.device)
        #     base_input.append(inactive_fraction)
        
        x = torch.cat(base_input, dim=1)
        
        logits = self.network(x)
        logits = torch.clamp(logits, -5, 5)
        base_influence = torch.sigmoid(logits / 2.0)
        
        connection_strength = link_repr[:, 1] + link_repr[:, 2]
        gated_influence = base_influence.squeeze() * connection_strength

        assert not torch.isnan(gated_influence).any(), \
            f"InfluenceNN NaN! logits range: [{logits.min():.3f}, {logits.max():.3f}]"
        
        return gated_influence.unsqueeze(-1)


class InteractionFormationNN(nn.Module):
    """NN_form: Input [f_ij(t), s_i(t), s_j(t), dist_ij] - unchanged"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        input_dim = feature_dim + 3 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1)
        # )
    
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
    """
    NN_self: Input [x_i(t), s_i(t:t-L+1)^{-k}, k, t, (optional: RBF_features, inactive_fraction)] 
    
    Enhanced features (when use_enhanced_features=True):
    - RBF temporal features (num_rbf_centers dim): Encode time distribution
    - Inactive fraction (1 dim): Fraction of population still inactive for this decision
    """
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64, num_layers: int = 2,
                 use_enhanced_features: bool = False, num_rbf_centers: int = 5, 
                 rbf_sigma: float = 3.0, max_time: int = 23):
        super().__init__()
        self.L = L
        self.num_layers = num_layers
        self.use_enhanced_features = use_enhanced_features
        self.num_rbf_centers = num_rbf_centers
        self.rbf_sigma = rbf_sigma
        self.max_time = max_time
        
        # Base input: features + state_history + decision_type + time
        base_input_dim = feature_dim + L * 2 + 3 + 1
        
        # Enhanced features (when enabled)
        enhanced_dim = 0
        if use_enhanced_features:
            # Only inactive_fraction now (RBF disabled)
            enhanced_dim = 1  # inactive_fraction only
            
        input_dim = base_input_dim + enhanced_dim
        
        # Register RBF centers as buffer (not trainable)
        if use_enhanced_features:
            # Distribute RBF centers across time span
            centers = torch.linspace(0, max_time, num_rbf_centers)
            self.register_buffer('rbf_centers', centers)
        
        # Build flexible multi-layer network
        layers = []
        current_dim = input_dim
        
        if num_layers == 2:
            # Original 2-layer architecture (backward compatible)
            layers.extend([
                nn.Linear(current_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ])
        elif num_layers == 3:
            # 3-layer architecture (medium capacity)
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ])
        elif num_layers == 4:
            # 4-layer architecture (high capacity)
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ])
        else:
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")
        
        self.network = nn.Sequential(*layers)
    
    def _compute_rbf_features(self, time: torch.Tensor) -> torch.Tensor:
        """Compute RBF (Gaussian) features for time encoding."""
        # time: [batch_size, 1]
        # rbf_centers: [num_rbf_centers]
        # Output: [batch_size, num_rbf_centers]
        batch_size = time.shape[0]
        time_expanded = time.expand(-1, self.num_rbf_centers)  # [batch_size, num_centers]
        centers_expanded = self.rbf_centers.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_centers]
        
        # Gaussian RBF: exp(-(t - center)^2 / (2 * sigma^2))
        rbf_features = torch.exp(-((time_expanded - centers_expanded) ** 2) / (2 * self.rbf_sigma ** 2))
        return rbf_features
    
    def _compute_time_decay(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute time decay weight for self-activation.
        As time progresses, remaining households are more 'stubborn', so decay self-activation.
        
        Uses exponential decay: exp(-t / tau)
        - tau = max_time / 3.0 → more aggressive decay
        - At t=max_time, decay ≈ 0.05 (95% reduction)
        """
        # More aggressive decay: tau = 23/3 ≈ 7.67
        # t=10: 0.26, t=15: 0.14, t=23: 0.05
        tau = self.max_time / 3.0  # Changed from /2.0 to /3.0 for stronger decay
        decay_weight = torch.exp(-time / tau)
        return decay_weight
    
    def forward(self, features, state_history_excluding_k, decision_type_onehot, time, 
                inactive_fraction=None):
        """
        Args:
            inactive_fraction: [batch_size, 1] - Fraction of population still inactive for this decision
                              Only used when use_enhanced_features=True
        """
        # Base features
        base_input = [features, state_history_excluding_k, decision_type_onehot, time]
        
        # ✅ Enhanced features DISABLED (inactive_fraction caused over-prediction)
        # if self.use_enhanced_features:
        #     # 1. RBF temporal encoding - DISABLED (too strong, causes time-only fitting)
        #     # rbf_features = self._compute_rbf_features(time)
        #     # base_input.append(rbf_features)
        #     
        #     # 2. Population saturation info (inactive fraction)
        #     if inactive_fraction is None:
        #         # Fallback: assume 50% inactive if not provided
        #         inactive_fraction = torch.full((features.shape[0], 1), 0.5, device=features.device)
        #     base_input.append(inactive_fraction)
        
        x = torch.cat(base_input, dim=1)
        output = torch.sigmoid(self.network(x))
        
        # ✅ Time decay and enhanced features DISABLED
        # Previously caused over-prediction issues
        # if self.use_enhanced_features:
        #     decay_weight = self._compute_time_decay(time)
        #     output = output * decay_weight
        
        assert not torch.isnan(output).any(), "SelfActivationNN output NaN!"
        return output



