import torch
import torch.nn as nn
import torch.nn.functional as F
from .neural_networks import InteractionFormationNN
from .utils import compute_pairwise_features


class NetworkEvolution(nn.Module):
    """Network evolution model with f_ij = |features_i - features_j| everywhere."""
    
    def __init__(self, interaction_nn: InteractionFormationNN):
        super().__init__()
        self.interaction_nn = interaction_nn
        
        # 只需要这一个learnable parameter！
        self.alpha_0 = nn.Parameter(torch.tensor(1.0))  # For bonding scaling in initial probabilities
        
        # Register as buffers (for similarity calculation)
        self.register_buffer('sigma_demo_sq', torch.tensor(1.0))
        self.register_buffer('sigma_geo_sq', torch.tensor(1.0))


        
    def set_normalization_factors(self, all_features: torch.Tensor, all_distances: torch.Tensor):
        """Precompute median normalization using f_ij = |features_i - features_j|"""
        n_households = all_features.shape[0]
        
        demo_dists = []
        geo_dists = []
        
        for i in range(n_households):
            for j in range(i + 1, n_households):
                # Use absolute difference for demographic distance
                f_ij = compute_pairwise_features(all_features[i], all_features[j])
                demo_dist = torch.sqrt(torch.sum(f_ij ** 2))
                demo_dists.append(demo_dist)
                geo_dists.append(all_distances[i, j])
        
        if len(demo_dists) > 0:
            # Update buffers with tensor values
            demo_median = torch.median(torch.stack(demo_dists)) ** 2
            geo_median = torch.median(torch.stack(geo_dists)) ** 2

            self.sigma_demo_sq.data = max(demo_median ** 2, torch.tensor(1e-6))
            self.sigma_geo_sq.data = max(geo_median ** 2, torch.tensor(1e-6))

        print(f"Normalization: σ_demo² = {self.sigma_demo_sq.item():.4f}, σ_geo² = {self.sigma_geo_sq.item():.4f}")

    def similarity(self, features_i, features_j, distances):
        """Similarity using f_ij = |features_i - features_j|"""
        f_ij = compute_pairwise_features(features_i, features_j)
        demo_dist_sq = torch.sum(f_ij ** 2, dim=1, keepdim=True)
        geo_dist_sq = distances ** 2
        
        return torch.exp(-demo_dist_sq / self.sigma_demo_sq.item() - geo_dist_sq / self.sigma_geo_sq.item())
    
    def interaction_potential(self, features_i, features_j, states_i, states_j, distances):
        """Interaction potential using f_ij = |features_i - features_j|"""
        return self.interaction_nn(features_i, features_j, states_i, states_j, distances)

    # def __init__(self, interaction_nn: InteractionFormationNN):
    #     super().__init__()
    #     self.interaction_nn = interaction_nn

    #     # # Learnable parameters
    #     # self.alpha_bonding = nn.Parameter(torch.tensor(0.005))
    #     # self.beta_form = nn.Parameter(torch.tensor(0.2))
    #     # self.gamma = nn.Parameter(torch.tensor(0.2))
    #     # self.alpha_0 = nn.Parameter(torch.tensor(0.1))
    #     # self.beta_0 = nn.Parameter(torch.tensor(0.2))
    #     self.alpha_bonding = nn.Parameter(torch.tensor(1.5))    # MUCH LARGER
    #     self.beta_form = nn.Parameter(torch.tensor(2.0))        # LARGER  
    #     self.gamma = nn.Parameter(torch.tensor(0.2))            # Keep
    #     self.alpha_0 = nn.Parameter(torch.tensor(1.0))          # LARGER
    #     self.beta_0 = nn.Parameter(torch.tensor(2.0)) 

    #     # Precomputed normalization factors
    #     self.sigma_demo_sq = 1.0
    #     self.sigma_geo_sq = 1.0

    # def set_normalization_factors(self, all_features: torch.Tensor, all_distances: torch.Tensor):
    #     """Precompute median normalization using f_ij = |features_i - features_j|"""
    #     n_households = all_features.shape[0]

    #     demo_dists = []
    #     geo_dists = []

    #     for i in range(n_households):
    #         for j in range(i + 1, n_households):
    #             # Use absolute difference for demographic distance
    #             f_ij = compute_pairwise_features(all_features[i], all_features[j])
    #             demo_dist = torch.sqrt(torch.sum(f_ij ** 2))
    #             demo_dists.append(demo_dist)
    #             geo_dists.append(all_distances[i, j])

    #     if len(demo_dists) > 0:
    #         self.sigma_demo_sq = (torch.median(torch.stack(demo_dists))**2).item()
    #         self.sigma_geo_sq = (torch.median(torch.stack(geo_dists))**2).item()

    #     print(f"Normalization: σ_demo² = {self.sigma_demo_sq:.4f}, σ_geo² = {self.sigma_geo_sq:.4f}")
    
    def transition_probabilities(self, prev_link_type, features_i, features_j, 
                               states_i, states_j, distances):
        """UPDATED: Add mobility constraints following PDF formulation"""
        batch_size = prev_link_type.shape[0]
        
        sim = self.similarity(features_i, features_j, distances)
        interaction_pot = self.interaction_potential(features_i, features_j, states_i, states_j, distances)
        
        trans_probs = torch.zeros((batch_size, 3, 3))
        
        # Check vacancy status
        vacant_i = states_i[:, 0:1]  # [batch_size, 1]
        vacant_j = states_j[:, 0:1]  # [batch_size, 1]
        either_vacant = (vacant_i + vacant_j).clamp(0, 1)  # 1 if either is vacant
        
        # From no connection (ℓ_ij(t-1) = 0)
        # If either vacant: p(0|0)=1, p(1|0)=0, p(2|0)=0
        # Otherwise: p(0|0)=1-interaction_pot, p(1|0)=0, p(2|0)=interaction_pot
        
        trans_probs[:, 0, 0] = torch.where(
            either_vacant.squeeze(1) > 0.5,
            torch.ones(batch_size),  # If either vacant: prob=1 for no link
            1 - interaction_pot.squeeze(1)  # Otherwise: 1-interaction_potential
        )
        
        trans_probs[:, 0, 1] = 1e-6  # Always 0 for bonding from no-link
        
        trans_probs[:, 0, 2] = torch.where(
            either_vacant.squeeze(1) > 0.5,
            torch.zeros(batch_size),  # If either vacant: prob=0 for bridging
            interaction_pot.squeeze(1)  # Otherwise: interaction_potential
        )
        
        # From bonding (ℓ_ij(t-1) = 1) - never disappear
        trans_probs[:, 1, 0] = 1e-6
        trans_probs[:, 1, 1] = 1.0 - 2e-6  # Always persist
        trans_probs[:, 1, 2] = 1e-6
        
        # From bridging (ℓ_ij(t-1) = 2)
        # If either vacant: p(0|2)=1-interaction_pot, p(1|2)=0, p(2|2)=interaction_pot
        # Otherwise: p(0|2)=0, p(1|2)=0, p(2|2)=1
        
        trans_probs[:, 2, 0] = torch.where(
            either_vacant.squeeze(1) > 0.5,
            1 - interaction_pot.squeeze(1),  # If either vacant: decay prob
            torch.zeros(batch_size)  # Otherwise: no decay
        )
        
        trans_probs[:, 2, 1] = 1e-6*torch.ones(batch_size)  # Never become bonding
        
        trans_probs[:, 2, 2] = torch.where(
            either_vacant.squeeze(1) > 0.5,
            interaction_pot.squeeze(1),  # If either vacant: stay prob
            torch.ones(batch_size)  # Otherwise: perfect persistence
        )

        row_sums = trans_probs.sum(dim=2, keepdim=True)
        trans_probs = trans_probs / row_sums
        
        return trans_probs
    
    def initial_probabilities(self, features_i, features_j, distances):
        """UPDATED: Use log formulation with alpha_0 for bonding"""
        sim = self.similarity(features_i, features_j, distances)
        
        batch_size = features_i.shape[0]
        zero_states = torch.zeros((batch_size, 3))
        interaction_pot = self.interaction_potential(features_i, features_j, zero_states, zero_states, distances)
        
        # Use the new log formulation
        logit_0 = torch.zeros_like(sim)                      # baseline = 0
        logit_1 = torch.log(self.alpha_0 * sim + 1e-8)      # log(alpha_0 * similarity + ε)
        logit_2 = torch.log(interaction_pot + 1e-8)          # log(interaction_potential + ε)

        logits = torch.stack([logit_0, logit_1, logit_2], dim=1)
        probabilities = F.softmax(logits, dim=1)

        return probabilities