import torch
import torch.nn as nn
import torch.nn.functional as F
from .neural_networks import InteractionFormationNN
from .utils import compute_pairwise_features


class NetworkEvolution(nn.Module):
    """Network evolution model aligned with PDF formulation."""
    
    def __init__(self, interaction_nn: InteractionFormationNN):
        super().__init__()
        self.interaction_nn = interaction_nn
        
        self.alpha_0 = nn.Parameter(torch.tensor(1.0))  # For bonding scaling in initial probabilities
        
        # Register as buffers (for similarity calculation)
        self.register_buffer('sigma_demo_sq', torch.tensor(1.0))
        self.register_buffer('sigma_geo_sq', torch.tensor(1.0))

    def set_normalization_factors(self, all_features: torch.Tensor, all_distances: torch.Tensor,
                                neighbor_index=None):
        n = all_features.shape[0]
        demo_dists, geo_dists = [], []

        if neighbor_index is None:
            # old behavior
            for i in range(n):
                for j in range(i+1, n):
                    f_ij = compute_pairwise_features(all_features[i], all_features[j])
                    demo_dists.append(torch.sqrt(torch.sum(f_ij ** 2)))
                    geo_dists.append(all_distances[i, j])
        else:
            # sparse behavior
            for i in range(n):
                for j in neighbor_index[i]:
                    if j <= i:
                        continue
                    f_ij = compute_pairwise_features(all_features[i], all_features[j])
                    demo_dists.append(torch.sqrt(torch.sum(f_ij ** 2)))
                    geo_dists.append(all_distances[i, j])

        if demo_dists:
            self.sigma_demo_sq.data = torch.median(torch.stack(demo_dists)) ** 2
            self.sigma_geo_sq.data  = torch.median(torch.stack(geo_dists)) ** 2

        print(f"Normalization: σ_demo²={self.sigma_demo_sq.item():.4f}, "
            f"σ_geo²={self.sigma_geo_sq.item():.4f}")

    def similarity(self, features_i, features_j, distances):
        """Similarity using f_ij = |features_i - features_j|"""
        f_ij = compute_pairwise_features(features_i, features_j)
        demo_dist_sq = torch.sum(f_ij ** 2, dim=1, keepdim=True)
        geo_dist_sq = distances ** 2
        
        return torch.exp(-demo_dist_sq / self.sigma_demo_sq.item() - geo_dist_sq / self.sigma_geo_sq.item())
    
    def interaction_potential(self, features_i, features_j, states_i, states_j, distances):
        """Interaction potential using f_ij = |features_i - features_j|"""
        return self.interaction_nn(features_i, features_j, states_i, states_j, distances)
    
    def get_static_transition_probs(self, states_i_t, states_j_t) -> torch.Tensor:
        """
        Get static transition probability matrix [3, 3].
        Only depends on whether either household is vacant.
        """
        trans = torch.zeros(3, 3)
        
        # No link: stays no link
        trans[0, 0] = 1.0 - 2e-8
        trans[0, 1:] = 1e-8
        
        # Bonding: persists forever
        trans[1, 1] = 1.0 - 2e-8
        trans[1, [0, 2]] = 1e-8
        
        # Bridging: breaks if either vacant, else persists
        either_vacant = (states_i_t[0] + states_j_t[0]).clamp(0, 1)
        if either_vacant > 0.5:
            trans[2, :] = torch.tensor([1.0 - 2e-8, 1e-8, 1e-8])  # Break
        else:
            trans[2, :] = torch.tensor([1e-8, 1e-8, 1.0 - 2e-8])  # Persist
        
        return trans
    
    def initial_probabilities(self, features_i, features_j, distances):
        """Initial probabilities at t=0 - UNCHANGED (already correct as prior)"""
        sim = self.similarity(features_i, features_j, distances)
        
        batch_size = features_i.shape[0]
        zero_states = torch.zeros((batch_size, 3))
        interaction_pot = self.interaction_potential(features_i, features_j, zero_states, zero_states, distances)
        
        # Use the log formulation for initial prior
        logit_0 = torch.zeros_like(sim)                      # baseline = 0
        logit_1 = torch.log(self.alpha_0 * sim + 1e-8)      # log(alpha_0 * similarity + ε)
        logit_2 = torch.log(interaction_pot + 1e-8)          # log(interaction_potential + ε)

        logits = torch.stack([logit_0, logit_1, logit_2], dim=1)
        probabilities = F.softmax(logits, dim=1)

        return probabilities