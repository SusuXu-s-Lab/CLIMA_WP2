import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from models import NetworkEvolution, StateTransition

class ELBOComputation:
    
    def __init__(self, network_evolution: NetworkEvolution, state_transition: StateTransition,
                 sparsity_weight: float = 0.0):
        self.network_evolution = network_evolution
        self.state_transition = state_transition
        self.sparsity_weight = sparsity_weight
        
        # Observation model parameters
        self.rho_1 = torch.nn.Parameter(torch.tensor(0.3))
        self.rho_2 = torch.nn.Parameter(torch.tensor(0.4))
    
    def compute_state_likelihood_batch(self,
                                     features: torch.Tensor,
                                     states: torch.Tensor,
                                     distances: torch.Tensor,
                                     node_batch: torch.Tensor,
                                     network_data,
                                     gumbel_samples: List[Dict[str, torch.Tensor]],
                                     max_timestep: int) -> torch.Tensor:
        """
        State likelihood using marginal-based Gumbel-Softmax samples.

        """
        total_likelihood = 0.0
        num_samples = len(gumbel_samples)
        
        for sample_idx, current_samples in enumerate(gumbel_samples):
            sample_likelihood = 0.0
            
            for t in range(max_timestep):
                for decision_k in range(3):
                    # Find undecided households in batch
                    batch_undecided = []
                    for node_idx in node_batch:
                        if states[node_idx, t, decision_k] == 0:
                            batch_undecided.append(node_idx)
                    
                    if len(batch_undecided) == 0:
                        continue
                    
                    batch_undecided_tensor = torch.tensor(batch_undecided, dtype=torch.long)
                    
                    # Compute activation probabilities (same as before)
                    activation_probs = self.state_transition.compute_activation_probability(
                        household_idx=batch_undecided_tensor,
                        decision_type=decision_k,
                        features=features,
                        states=states,
                        distances=distances,
                        network_data=network_data,
                        gumbel_samples=current_samples,
                        time=t
                    )
                    
                    # Get actual outcomes
                    actual_outcomes = states[batch_undecided, t+1, decision_k]
                    
                    # Compute log likelihood
                    log_probs = actual_outcomes * torch.log(activation_probs + 1e-8) + \
                               (1 - actual_outcomes) * torch.log(1 - activation_probs + 1e-8)
                    
                    sample_likelihood += torch.sum(log_probs)
            
            total_likelihood += sample_likelihood
        
        return total_likelihood / num_samples
    
    def compute_network_observation_likelihood_batch(self,
                                                   marginal_probs: Dict[str, torch.Tensor],
                                                   node_batch: torch.Tensor,
                                                   network_data,
                                                   max_timestep: int) -> torch.Tensor:
        """
        Network observation likelihood using MARGINAL probabilities.
        """
        total_likelihood = 0.0
        batch_nodes_set = set(node_batch.tolist())
        
        for t in range(max_timestep + 1):
            # Observed links involving batch nodes
            observed_edges = network_data.get_observed_edges_at_time(t)
            for edge in observed_edges:
                i, j, link_type = edge[0], edge[1], edge[2]
                
                if i in batch_nodes_set or j in batch_nodes_set:
                    if link_type == 1:
                        obs_prob = 1 - self.rho_1
                    elif link_type == 2:
                        obs_prob = 1 - self.rho_2
                    else:
                        obs_prob = 1.0
                    
                    total_likelihood += torch.log(obs_prob + 1e-8)
            
            # Hidden links involving batch nodes - use MARGINAL probabilities
            hidden_pairs = network_data.get_hidden_pairs(t)
            for i, j in hidden_pairs:
                if i in batch_nodes_set or j in batch_nodes_set:
                    pair_key = f"{i}_{j}_{t}"
                    if pair_key in marginal_probs:
                        π̄_ij = marginal_probs[pair_key]  # Use marginal instead of categorical
                        
                        missing_prob = (π̄_ij[0] * 1.0 +
                                       π̄_ij[1] * self.rho_1 +
                                       π̄_ij[2] * self.rho_2)
                        
                        total_likelihood += torch.log(missing_prob + 1e-8)
        
        return total_likelihood  # Will be normalized in the caller
    
    def compute_prior_likelihood_batch(self,
                                     conditional_probs: Dict[str, torch.Tensor],
                                     marginal_probs: Dict[str, torch.Tensor],
                                     features: torch.Tensor,
                                     states: torch.Tensor,
                                     distances: torch.Tensor,
                                     node_batch: torch.Tensor,
                                     network_data,
                                     max_timestep: int) -> torch.Tensor:
        """
        Prior likelihood using both marginals and conditionals.
        """
        total_likelihood = 0.0
        batch_nodes_set = set(node_batch.tolist())
        
        # Initial prior (t=0)
        for i, j in network_data.get_hidden_pairs(0):
            if i in batch_nodes_set or j in batch_nodes_set:
                pair_key = f"{i}_{j}_0"
                if pair_key in marginal_probs:
                    π̄_ij = marginal_probs[pair_key]  # Use marginal for initial
                    
                    # Compute initial probabilities from network evolution model
                    feat_i = features[i].unsqueeze(0)
                    feat_j = features[j].unsqueeze(0)
                    dist = distances[i, j].unsqueeze(0).unsqueeze(1)
                    
                    init_probs = self.network_evolution.initial_probabilities(feat_i, feat_j, dist)
                    
                    # Σ π_ij(0)[k] × log p(ℓ_ij(0) = k | θ, ψ)
                    log_prob = torch.sum(π̄_ij * torch.log(init_probs.squeeze(0) + 1e-8))
                    total_likelihood += log_prob
        
        # Temporal transitions (t >= 1) - this is the complex part
        for t in range(1, max_timestep + 1):
            for i, j in network_data.get_hidden_pairs(t):
                if i in batch_nodes_set or j in batch_nodes_set:
                    pair_key_current = f"{i}_{j}_{t}"
                    pair_key_prev = f"{i}_{j}_{t-1}"
                    
                    if pair_key_current in conditional_probs and pair_key_current in marginal_probs:
                        π_conditional = conditional_probs[pair_key_current]  # [3, 3] - π_ij(t | k')
                        π_current = marginal_probs[pair_key_current]  # [3] - π̄_ij(t)
                        
                        # Get previous probabilities π̄_ij^prev(t-1)
                        if network_data.is_observed(i, j, t-1):
                            # Previous state observed: use one-hot
                            prev_type = network_data.get_link_type(i, j, t-1)
                            π_prev = F.one_hot(torch.tensor(prev_type), num_classes=3).float()
                        elif pair_key_prev in marginal_probs:
                            # Previous state hidden: use previous marginal
                            π_prev = marginal_probs[pair_key_prev]
                        else:
                            # Skip if we don't have previous information
                            continue
                        
                        # Compute transition probabilities from network evolution model
                        feat_i = features[i].unsqueeze(0)
                        feat_j = features[j].unsqueeze(0)
                        state_i = states[i, t].unsqueeze(0)
                        state_j = states[j, t].unsqueeze(0)
                        dist = distances[i, j].unsqueeze(0).unsqueeze(1)
                        
                        # Need to expand prev state for network evolution model
                        π_prev_expanded = π_prev.unsqueeze(0)  # [1, 3]
                        trans_probs = self.network_evolution.transition_probabilities(
                            π_prev_expanded, feat_i, feat_j, state_i, state_j, dist
                        )  # [1, 3, 3]
                        trans_probs = trans_probs.squeeze(0)  # [3, 3]
                        
                        # Complex formula: Σ Σ π̄_ij^prev(t-1)[k'] × π_ij(t | k')[k] × log p(k|k', S_t, θ, ψ)
                        for k_prev in range(3):
                            for k_curr in range(3):
                                prob_contrib = (π_prev[k_prev] * 
                                              π_conditional[k_prev, k_curr] * 
                                              torch.log(trans_probs[k_prev, k_curr] + 1e-8))
                                total_likelihood += prob_contrib
        
        # NORMALIZATION ADDED: Divide by total possible link-timesteps
        total_households = features.shape[0]
        total_possible_pairs = total_households * (total_households - 1) // 2
        total_possible_link_timesteps = total_possible_pairs * (max_timestep + 1)
        
        return total_likelihood 
    
    def compute_posterior_entropy_batch(self, 
                                       conditional_probs: Dict[str, torch.Tensor],
                                       marginal_probs: Dict[str, torch.Tensor],
                                       network_data,
                                       node_batch: torch.Tensor,
                                       max_timestep: int) -> torch.Tensor:
        """
        Posterior entropy using conditional entropy formulas from PDF.
        """
        total_entropy = 0.0
        batch_nodes_set = set(node_batch.tolist())
        
        # Initial entropy (t=0) - simple case
        for i, j in network_data.get_hidden_pairs(0):
            if i in batch_nodes_set or j in batch_nodes_set:
                pair_key = f"{i}_{j}_0"
                if pair_key in marginal_probs:
                    π̄_ij = marginal_probs[pair_key]
                    entropy = -torch.sum(π̄_ij * torch.log(π̄_ij + 1e-8))
                    total_entropy += entropy
        
        # Temporal entropy (t >= 1) - conditional entropy formulas
        for t in range(1, max_timestep + 1):
            for i, j in network_data.get_hidden_pairs(t):
                if i in batch_nodes_set or j in batch_nodes_set:
                    pair_key_current = f"{i}_{j}_{t}"
                    pair_key_prev = f"{i}_{j}_{t-1}"
                    
                    if pair_key_current in conditional_probs:
                        π_conditional = conditional_probs[pair_key_current]  # [3, 3]
                        
                        if network_data.is_observed(i, j, t-1):
                            # Previous state observed: H = -Σ π_ij(t | ℓ_obs)[k] × log π_ij(t | ℓ_obs)[k]
                            prev_type = network_data.get_link_type(i, j, t-1)
                            conditional_given_obs = π_conditional[prev_type, :]  # [3]
                            entropy = -torch.sum(conditional_given_obs * torch.log(conditional_given_obs + 1e-8))
                            
                        elif pair_key_prev in marginal_probs:
                            # Previous state hidden: H = -Σ Σ [π̄_ij(t-1)[k'] × π_ij(t | k')[k]] × log π_ij(t | k')[k]
                            π_prev = marginal_probs[pair_key_prev]  # [3]
                            entropy = 0.0
                            
                            for k_prev in range(3):
                                for k_curr in range(3):
                                    joint_prob = π_prev[k_prev] * π_conditional[k_prev, k_curr]
                                    if joint_prob > 1e-8:
                                        entropy -= joint_prob * torch.log(π_conditional[k_prev, k_curr] + 1e-8)
                        else:
                            # Skip if we don't have previous information
                            continue
                        
                        total_entropy += entropy
        
        # NORMALIZATION ADDED: We need to get total_households from somewhere
        # We'll add it to the method signature in the main compute_elbo_batch method
        return total_entropy  # Will be normalized in the caller
    
    def compute_sparsity_regularization(self, marginal_probs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Network sparsity regularization using marginal probabilities.
        """
        if self.sparsity_weight == 0.0:
            return torch.tensor(0.0)
        
        total_entropy = 0.0
        for pair_key, π̄_ij in marginal_probs.items():
            entropy = -torch.sum(π̄_ij * torch.log(π̄_ij + 1e-8))
            total_entropy += entropy
        
        # NORMALIZATION: Will be applied in the caller
        return total_entropy  # Raw entropy, will be normalized in compute_elbo_batch
    

    def _get_activation_prob(self, node_idx, decision_type, features, states, distances, 
                        network_data, gumbel_samples, time):
        """Helper to get activation probability for a single household"""
        return self.state_transition.compute_activation_probability(
            household_idx=torch.tensor([node_idx], dtype=torch.long),
            decision_type=decision_type,
            features=features,
            states=states,
            distances=distances,
            network_data=network_data,
            gumbel_samples=gumbel_samples,
            time=time
        )[0]
    
    def compute_constraint_penalty(self, features, states, distances, node_batch, 
                              network_data, gumbel_samples, max_timestep):
        """
        NEW: Compute decision constraint penalties:
        1. repair vs (vacant + sell) - can't repair if moved/sold
        2. sell implies vacant - if sold house, should have moved out
        """
        if len(gumbel_samples) == 0:
            return torch.tensor(0.0)
        
        current_samples = gumbel_samples[0]
        total_penalty = 0.0
        count = 0
        
        for t in range(max_timestep + 1):
            for node_idx in node_batch:
                # Get current observed states
                vacant_obs = states[node_idx, t, 0]  # 0 or 1
                repair_obs = states[node_idx, t, 1]  # 0 or 1  
                sell_obs = states[node_idx, t, 2]    # 0 or 1
                
                penalty_1 = 0.0
                penalty_2 = 0.0
                
                # Penalty 1: repair vs (vacant + sell) - only if repair=0
                if repair_obs == 0:
                    # Get predicted probabilities
                    vacant_prob = vacant_obs if vacant_obs == 1 else self._get_activation_prob(
                        node_idx, 0, features, states, distances, network_data, current_samples, t)
                    
                    repair_prob = self._get_activation_prob(
                        node_idx, 1, features, states, distances, network_data, current_samples, t)
                    
                    sell_prob = sell_obs if sell_obs == 1 else self._get_activation_prob(
                        node_idx, 2, features, states, distances, network_data, current_samples, t)
                    
                    # Can't repair if moved out or sold house
                    penalty_1 = repair_prob * (vacant_prob + sell_prob)
                
                # Penalty 2: sell implies vacant - only if vacant=0 and sell=1
                # Cases: [0,0,1] and [0,1,1]
                if vacant_obs == 0 and sell_obs == 1:
                    # If sold but not moved out, that's illogical
                    vacant_prob = self._get_activation_prob(
                        node_idx, 0, features, states, distances, network_data, current_samples, t)
                    
                    penalty_2 = 1 - vacant_prob  # Penalty if vacant_prob is low when sell=1
                
                total_penalty += (penalty_1 + penalty_2)
                count += 1
        
        return total_penalty / count if count > 0 else torch.tensor(0.0)
    
    def compute_elbo_batch(self,
                        features: torch.Tensor,
                        states: torch.Tensor,
                        distances: torch.Tensor,
                        node_batch: torch.Tensor,
                        network_data,
                        conditional_probs: Dict[str, torch.Tensor],
                        marginal_probs: Dict[str, torch.Tensor],
                        gumbel_samples: List[Dict[str, torch.Tensor]],
                        max_timestep: int,
                        lambda_constraint: float = 0.01) -> Dict[str, torch.Tensor]:        
        """
        Complete ELBO computation with per-example normalization and component weighting.
        """
        
        # Get normalization factors
        total_households = features.shape[0]
        total_possible_pairs = total_households * (total_households - 1) // 2
        total_timesteps = max_timestep + 1
        
        # Compute raw likelihoods (without any normalization first)
        state_likelihood_raw = self.compute_state_likelihood_batch(
            features, states, distances, node_batch, network_data, gumbel_samples, max_timestep
        )
        observation_likelihood_raw = self.compute_network_observation_likelihood_batch(
            marginal_probs, node_batch, network_data, max_timestep
        )
        prior_likelihood_raw = self.compute_prior_likelihood_batch(
            conditional_probs, marginal_probs, features, states, distances, 
            node_batch, network_data, max_timestep
        )
        posterior_entropy_raw = self.compute_posterior_entropy_batch(
            conditional_probs, marginal_probs, network_data, node_batch, max_timestep
        )
        sparsity_reg_raw = self.compute_sparsity_regularization(marginal_probs)

        constraint_penalty_raw = self.compute_constraint_penalty(
        features, states, distances, node_batch, network_data, 
        gumbel_samples, max_timestep
        )
        
        print(f"Raw values - State: {state_likelihood_raw.item():.2f}, "
            f"Obs: {observation_likelihood_raw.item():.2f}, "
            f"Prior: {prior_likelihood_raw.item():.2f}, "
            f"Entropy: {posterior_entropy_raw.item():.2f},"
            f" Sparsity: {sparsity_reg_raw.item():.2f}, "
            f"Constraint: {constraint_penalty_raw.item():.2f}")
        
        # Calculate actual counts for proper normalization
        total_state_predictions = total_households * max_timestep * 3  # 3 decision types
        total_network_pairs = len(marginal_probs)  # Actual pairs being evaluated
        
        # Normalize by actual counts
        state_likelihood_per_prediction = state_likelihood_raw / total_state_predictions
        network_likelihood_per_pair = observation_likelihood_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        prior_likelihood_per_pair = prior_likelihood_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        entropy_per_pair = posterior_entropy_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        sparsity_per_pair = self.sparsity_weight * sparsity_reg_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        
        print(f"Per-unit values - State: {state_likelihood_per_prediction.item():.4f}, "
            f"Obs: {network_likelihood_per_pair.item():.4f}, "
            f"Prior: {prior_likelihood_per_pair.item():.4f}, "
            f"Entropy: {entropy_per_pair.item():.4f}")
        
        # Adaptive weighting based on magnitudes
        state_magnitude = abs(state_likelihood_per_prediction.item())
        network_magnitude = max(abs(network_likelihood_per_pair.item()), 
                            abs(prior_likelihood_per_pair.item()), 
                            abs(entropy_per_pair.item()))
        
        if network_magnitude > 0:
            state_weight = network_magnitude / state_magnitude if state_magnitude > 0 else 1.0
            state_weight = min(state_weight, 1.0)  # Cap at 1.0
        else:
            state_weight = 0.01  # Fallback small weight
        
        print(f"Adaptive state weight: {state_weight:.6f}")
        
        # Apply adaptive weighting
        weighted_state_likelihood = state_weight * state_likelihood_per_prediction
        weighted_observation_likelihood = 1.0 * network_likelihood_per_pair
        weighted_prior_likelihood = 1.0 * prior_likelihood_per_pair
        weighted_posterior_entropy = 1.0 * entropy_per_pair
        weighted_sparsity_reg = 1.0 * sparsity_per_pair
        weighted_constraint_penalty = -lambda_constraint * constraint_penalty_raw
        
        # Total weighted ELBO  
        total_elbo = (weighted_state_likelihood + weighted_observation_likelihood + 
                    weighted_prior_likelihood + weighted_posterior_entropy - weighted_sparsity_reg - weighted_constraint_penalty)
        
        print(f"Weighted components - State: {weighted_state_likelihood.item():.4f}, "
            f"Obs: {weighted_observation_likelihood.item():.4f}, "
            f"Prior: {weighted_prior_likelihood.item():.4f}, "
            f"Entropy: {weighted_posterior_entropy.item():.4f}")
        print(f"Sparsity: {weighted_sparsity_reg.item():.4f}, "
            f"Constraint: {weighted_constraint_penalty.item():.4f}")
        print(f"Total weighted ELBO: {total_elbo.item():.4f}")
        
        return {
            'state_likelihood': weighted_state_likelihood,
            'observation_likelihood': weighted_observation_likelihood, 
            'prior_likelihood': weighted_prior_likelihood,
            'posterior_entropy': weighted_posterior_entropy,
            'sparsity_regularization': weighted_sparsity_reg,
            'constraint_penalty': weighted_constraint_penalty,
            'total_elbo': total_elbo
        }