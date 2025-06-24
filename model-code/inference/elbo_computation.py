import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from models import NetworkEvolution, StateTransition

class ELBOComputation:
    """
    Updated ELBO computation following PDF formulation with normalization added.
    
    Key changes:
    1. Network observation likelihood uses marginal probabilities
    2. Prior likelihood uses both marginals and conditionals (much more complex)
    3. Posterior entropy uses conditional entropy formulas
    4. State likelihood uses marginal-based sampling (conceptually same)
    5. ALL TERMS NOW NORMALIZED BY GLOBAL PROBLEM SIZE
    """
    
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
        NOW WITH NORMALIZATION: Divided by total possible decision opportunities.
        """

        def compute_dynamic_class_weights(current_activations):
            n_positive = current_activations.sum()
            n_total = len(current_activations)
            
            if n_positive == 0:
                return 1.0, 1.0  
            
            pos_weight = min((n_total - n_positive) / n_positive, 10.0)
            return pos_weight, 1.0

        total_likelihood = 0.0
        num_samples = len(gumbel_samples[0])
        # print(f"Number of Gumbel samples: {num_samples}")
        
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

                    pos_weight, neg_weight = compute_dynamic_class_weights(actual_outcomes)
                    
                    # Compute log likelihood
                    pos_log_probs = actual_outcomes * torch.log(activation_probs + 1e-8)
                    neg_log_probs = (1 - actual_outcomes) * torch.log(1 - activation_probs + 1e-8)
                    
                    weighted_likelihood = (pos_weight * pos_log_probs.sum() + 
                                        neg_weight * neg_log_probs.sum())
                    sample_likelihood += weighted_likelihood
            
            total_likelihood += sample_likelihood
        
        return total_likelihood
    

    def compute_network_observation_likelihood_batch(self,
                                                marginal_probs: Dict[str, torch.Tensor],
                                                node_batch: torch.Tensor,
                                                network_data,
                                                max_timestep: int) -> torch.Tensor:
            """
            Network observation likelihood using MARGINAL probabilities.
            NOW WITH NORMALIZATION: Divided by total possible link-timesteps.
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
                            
                            missing_prob = (π̄_ij[0] * torch.log(torch.tensor(1.0)) +
                                        π̄_ij[1] * torch.log(self.rho_1 + 1e-8) +
                                        π̄_ij[2] * torch.log(self.rho_2 + 1e-8))     ### fixed
                            
                            total_likelihood += missing_prob
            
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
            FIXED: Prior likelihood for COMPLETE graph (observed + hidden).
            """
            total_likelihood = 0.0
            batch_nodes_set = set(node_batch.tolist())
            
            # Get ALL pairs involving batch nodes (not just hidden!)
            def get_all_batch_pairs(t):
                all_pairs = []
                for i in range(features.shape[0]):
                    for j in range(i + 1, features.shape[0]):
                        if i in batch_nodes_set or j in batch_nodes_set:
                            all_pairs.append((i, j))
                return all_pairs
            
            # Initial prior (t=0) - same formula for all pairs
            # print(f"all batch pairs at t=0: {get_all_batch_pairs(0)}")
            for i, j in get_all_batch_pairs(0):
                # Get probabilities (observed=one-hot, hidden=marginal)
                if network_data.is_observed(i, j, 0):
                    # Observed: use one-hot encoding
                    observed_type = network_data.get_link_type(i, j, 0)
                    π_ij = F.one_hot(torch.tensor(observed_type), num_classes=3).float()
                else:
                    # Hidden: use marginal probabilities
                    pair_key = f"{i}_{j}_0"
                    if pair_key in marginal_probs:
                        π_ij = marginal_probs[pair_key]
                    else:
                        continue  # Skip if no marginal available
                
                # Compute initial probabilities from network evolution model
                feat_i = features[i].unsqueeze(0)
                feat_j = features[j].unsqueeze(0)
                dist = distances[i, j].unsqueeze(0).unsqueeze(1)
                
                init_probs = self.network_evolution.initial_probabilities(feat_i, feat_j, dist)
                
                # Σ π_ij(0)[k] × log p(ℓ_ij(0) = k | θ, ψ)
                # print(f"i,j: {i},{j}, π_ij: {π_ij}, init_probs: {init_probs.squeeze(0)}")
                log_prob = torch.sum(π_ij * torch.log(init_probs.squeeze(0) + 1e-8))
                # print(f"Initial prior for pair {i},{j} at t=0: {log_prob.item()}")
                total_likelihood += log_prob
            
            # Temporal transitions (t >= 1) - handle all 4 cases
            for t in range(1, max_timestep + 1):
                for i, j in get_all_batch_pairs(t):
                    # Determine current and previous observation status
                    current_observed = network_data.is_observed(i, j, t)
                    prev_observed = network_data.is_observed(i, j, t-1)
                    
                    # Get transition probabilities from network evolution model
                    feat_i = features[i].unsqueeze(0)
                    feat_j = features[j].unsqueeze(0)
                    state_i = states[i, t].unsqueeze(0)
                    state_j = states[j, t].unsqueeze(0)
                    dist = distances[i, j].unsqueeze(0).unsqueeze(1)
                    
                    if not current_observed and not prev_observed:
                        # Case 1: Hidden → Hidden (existing implementation)
                        pair_key_current = f"{i}_{j}_{t}"
                        pair_key_prev = f"{i}_{j}_{t-1}"
                        
                        if (pair_key_current in conditional_probs and 
                            pair_key_current in marginal_probs and 
                            pair_key_prev in marginal_probs):
                            
                            π_conditional = conditional_probs[pair_key_current]  # [3, 3]
                            π_prev = marginal_probs[pair_key_prev]  # [3]
                            
                            # Get transition probabilities for all previous types
                            π_prev_expanded = π_prev.unsqueeze(0)  # [1, 3]
                            trans_probs = self.network_evolution.transition_probabilities(
                                π_prev_expanded, feat_i, feat_j, state_i, state_j, dist
                            ).squeeze(0)  # [3, 3]
                            
                            # Complex formula: Σ Σ π̄_ij^prev(t-1)[k'] × π_ij(t | k')[k] × log p(k|k', S_t, θ, ψ)
                            for k_prev in range(3):
                                for k_curr in range(3):
                                    prob_contrib = (π_prev[k_prev] * 
                                                π_conditional[k_prev, k_curr] * 
                                                torch.log(trans_probs[k_prev, k_curr] + 1e-8))
                                    total_likelihood += prob_contrib
                    
                    elif current_observed and not prev_observed:
                        # Case 2: Hidden → Observed
                        pair_key_prev = f"{i}_{j}_{t-1}"
                        if pair_key_prev in marginal_probs:
                            π_prev = marginal_probs[pair_key_prev]  # [3]
                            k_obs = network_data.get_link_type(i, j, t)
                            
                            # Get transition probabilities for all 3 previous types
                            # Use identity matrix to represent all possible previous types
                            prev_types_all = torch.eye(3)  # [3, 3] - each row is one-hot for k'=0,1,2
                            
                            trans_probs = self.network_evolution.transition_probabilities(
                                prev_types_all, 
                                feat_i.expand(3, -1),  # Expand to match batch size
                                feat_j.expand(3, -1), 
                                state_i.expand(3, -1), 
                                state_j.expand(3, -1), 
                                dist.expand(3, -1)
                            )  # [3, 3, 3]
                            
                            # Extract p(k_obs | k_prev) for k_prev = 0, 1, 2
                            # trans_probs[i, i, k_obs] gives p(k_obs | k_prev=i)
                            transition_to_obs = torch.tensor([
                                trans_probs[0, 0, k_obs],  # p(k_obs | k_prev=0)
                                trans_probs[1, 1, k_obs],  # p(k_obs | k_prev=1) 
                                trans_probs[2, 2, k_obs]   # p(k_obs | k_prev=2)
                            ])
                            
                            # Compute: Σ π̄_ij^prev(t-1)[k'] × log p(k_obs|k', S_t, θ, ψ)
                            log_probs = torch.log(transition_to_obs + 1e-8)
                            prob_contrib = torch.sum(π_prev * log_probs)
                            total_likelihood += prob_contrib
                    
                    elif not current_observed and prev_observed:
                        # Case 3: Observed → Hidden
                        pair_key_current = f"{i}_{j}_{t}"
                        if pair_key_current in marginal_probs:
                            π_current = marginal_probs[pair_key_current]  # [3]
                            k_prev_obs = network_data.get_link_type(i, j, t-1)
                            
                            # Get transition probabilities from observed previous type
                            prev_type_tensor = torch.tensor([[k_prev_obs]], dtype=torch.long)
                            prev_onehot = F.one_hot(prev_type_tensor, num_classes=3).float()
                            prev_onehot = prev_onehot.squeeze(1)  # [1, 3]
                            
                            trans_probs = self.network_evolution.transition_probabilities(
                                prev_onehot, feat_i, feat_j, state_i, state_j, dist
                            )  # [1, 3, 3]
                            
                            # Sum over current hidden types
                            for k_curr in range(3):
                                trans_prob = trans_probs[0, k_prev_obs, k_curr]  # p(k_curr | k_prev_obs, ...)
                                prob_contrib = π_current[k_curr] * torch.log(trans_prob + 1e-8)
                                total_likelihood += prob_contrib
                    
                    else:
                        # Case 4: Observed → Observed
                        k_prev_obs = network_data.get_link_type(i, j, t-1)
                        k_curr_obs = network_data.get_link_type(i, j, t)
                        
                        # Get direct transition probability
                        prev_type_tensor = torch.tensor([[k_prev_obs]], dtype=torch.long)
                        prev_onehot = F.one_hot(prev_type_tensor, num_classes=3).float()
                        prev_onehot = prev_onehot.squeeze(1)  # [1, 3]
                        
                        trans_probs = self.network_evolution.transition_probabilities(
                            prev_onehot, feat_i, feat_j, state_i, state_j, dist
                        )  # [1, 3, 3]
                        
                        prob_contrib = trans_probs[0, k_prev_obs, k_curr_obs]
                        total_likelihood += torch.log(prob_contrib + 1e-8)
            
                    # print(f"i,j: {i},{j}, t: {t}, prob_contrib: {prob_contrib.item()}")
            
            return total_likelihood


    
    def compute_posterior_entropy_batch(self, 
                                       conditional_probs: Dict[str, torch.Tensor],
                                       marginal_probs: Dict[str, torch.Tensor],
                                       network_data,
                                       node_batch: torch.Tensor,
                                       max_timestep: int) -> torch.Tensor:
        """
        Posterior entropy using conditional entropy formulas from PDF.
        NOW WITH NORMALIZATION: Divided by total possible link-timesteps.
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
        NOW WITH NORMALIZATION: Divided by total possible link-timesteps.
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
        
        total_penalty = 0.0
        
        for sample_idx, current_samples in enumerate(gumbel_samples):
            sample_penalty = 0.0
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
                        # Get predicted probabilities using current sample
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
                    
                    sample_penalty += (penalty_1 + penalty_2)
                    count += 1
            
            if count > 0:
                total_penalty += sample_penalty / count
        
        return total_penalty / len(gumbel_samples)
    

    # def compute_connection_density_bonus(self, marginal_probs: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     """Reward network connectivity to counteract sparsity bias"""
    #     if len(marginal_probs) == 0:
    #         return torch.tensor(0.0)
        
    #     total_connection_prob = 0.0
    #     total_pairs = 0
        
    #     for pair_key, π_ij in marginal_probs.items():
    #         # connection probability is sum of probabilities for connected states
    #         connection_prob = π_ij[1] + π_ij[2]  
    #         total_connection_prob += connection_prob
    #         total_pairs += 1
        
    #     if total_pairs > 0:
    #         avg_connection_prob = total_connection_prob / total_pairs
    #         # reward higher connection density
    #         density_bonus = torch.log(avg_connection_prob + 1e-8)
    #         return density_bonus
        
    #     return torch.tensor(0.0)

    def compute_connection_density_bonus(self, marginal_probs, target_density=0.35):
        """Target-based density bonus instead of always rewarding connections"""
        
        connection_probs = []
        for pair_key, π_ij in marginal_probs.items():
            connection_prob = π_ij[1] + π_ij[2]
            connection_probs.append(connection_prob)
        
        if len(connection_probs) > 0:
            current_density = torch.mean(torch.stack(connection_probs))
            
            # If current density is below target, reward based on log difference
            if current_density < target_density:
                bonus = torch.log(current_density / target_density + 1e-8)  
            else:
                bonus = - (current_density - target_density) ** 2      
            
            return bonus
        
        return torch.tensor(0.0)
    
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
                        lambda_constraint: float = 0.01,
                        current_epoch: int = 0) -> Dict[str, torch.Tensor]:        
        """
        Complete ELBO computation with per-example normalization and component weighting.
        """

        # def get_dynamic_weights(epoch):
        #     if epoch < 30:  
        #         return {
        #             'state': 1.0, 'observation': 1.0, 'prior': 0.001,          
        #             'entropy': 1.0, 'sparsity': 0, 'constraint': -lambda_constraint, 'density_bonus': 1.0
        #         }
        #     elif epoch < 100:  # 50-100 epoch: introduce prior gradually
        #         progress = (epoch - 30) / 70  # 0 to 1
        #         prior_weight = 0.001 + 0.099 * progress
        #         return {
        #             'state': 1.0, 'observation': 1.0, 'prior': prior_weight ,
        #             'entropy': 1.0 - 0.3 * progress, 'sparsity': 0, 'constraint': -lambda_constraint, 'density_bonus': 1.0
        #         }
        #     else:  # 100 epoch: full weight
        #         return {
        #             'state': 1.0, 'observation': 1.0, 'prior': 0.1,
        #             'entropy': 0.7, 'sparsity': 0, 'constraint': -lambda_constraint, 'density_bonus': 1.0
        #         }

        def get_dynamic_weights(epoch):
            if epoch < 100:  
                return {
                    'state': 1.0, 
                    'observation': 1.0, 
                    'prior': 0.1,                          # 提高prior权重
                    'entropy': 1.0, 
                    'sparsity': 0, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 2.0                    # 降低density bonus
                }
            elif epoch < 200:
                progress = (epoch - 100) / 100
                prior_weight = 0.1 + 0.1 * progress      # 从0.1到0.2
                # density_weight = 2.0 * (1 - 0.5 * progress) # 从0.5降到0.25
                density_weight = 2.0
                return {
                    'state': 1.0, 
                    'observation': 1.0, 
                    'prior': prior_weight,
                    'entropy': 1.0 - 0.3 * progress, 
                    'sparsity': 0, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': density_weight
                }
            else:
                return {
                    'state': 1.0, 
                    'observation': 1.0, 
                    'prior': 0.2,
                    'entropy': 0.7, 
                    'sparsity': 0, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 2.0                   # 最终很小的bonus
                }

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

        density_bonus_raw = self.compute_connection_density_bonus(marginal_probs)

        print(f"Raw values - State: {state_likelihood_raw.item():.2f}, "
            f"Obs: {observation_likelihood_raw.item():.2f}, "
            f"Prior: {prior_likelihood_raw.item():.2f}, "
            f"Entropy: {posterior_entropy_raw.item():.2f},"
            f" Sparsity: {sparsity_reg_raw.item():.2f}, "
            f"Constraint: {constraint_penalty_raw.item():.2f}, "
            f"Density Bonus: {density_bonus_raw.item():.2f}")

        # Calculate actual counts for proper normalization
        # total_state_predictions =  max_timestep * 3  # 3 decision types
        total_state_predictions = len(gumbel_samples[0]) * len(gumbel_samples)  # 3 decision types per sample
        total_network_pairs = len(marginal_probs)  # Actual pairs being evaluated
        print(f"Total state predictions: {total_state_predictions}, Total network pairs: {total_network_pairs}")
        
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
        

        weight = get_dynamic_weights(current_epoch)
        
        # Apply adaptive weighting
        weighted_state_likelihood = weight['state'] * state_likelihood_per_prediction
        weighted_observation_likelihood = weight['observation'] * network_likelihood_per_pair
        weighted_prior_likelihood = weight['prior'] * prior_likelihood_per_pair
        weighted_posterior_entropy = weight['entropy'] * entropy_per_pair
        weighted_sparsity_reg = weight['sparsity'] * sparsity_per_pair
        weighted_constraint_penalty = weight['constraint'] * constraint_penalty_raw
        weighted_density_bonus = weight['density_bonus'] * density_bonus_raw
        
        # Total weighted ELBO  
        # total_elbo = (weighted_state_likelihood + weighted_observation_likelihood + 
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_sparsity_reg - weighted_constraint_penalty)
        total_elbo = (weighted_state_likelihood + weighted_observation_likelihood + 
                    weighted_prior_likelihood + weighted_posterior_entropy - weighted_sparsity_reg - weighted_constraint_penalty + weighted_density_bonus)
        
        print(f"Weighted components - State: {weighted_state_likelihood.item():.4f}, "
            f"Obs: {weighted_observation_likelihood.item():.4f}, "
            f"Prior: {weighted_prior_likelihood.item():.4f}, "
            f"Entropy: {weighted_posterior_entropy.item():.4f}")
        print(f"Sparsity: {weighted_sparsity_reg.item():.4f}, "
            f"Constraint: {weighted_constraint_penalty.item():.4f}, "
            f"Density Bonus: {weighted_density_bonus.item():.4f}")
        print(f"Total weighted ELBO: {total_elbo.item():.4f}")

        return {
            'state_likelihood': weighted_state_likelihood,
            'observation_likelihood': weighted_observation_likelihood, 
            'prior_likelihood': weighted_prior_likelihood,
            'posterior_entropy': weighted_posterior_entropy,
            'sparsity_regularization': weighted_sparsity_reg,
            'constraint_penalty': weighted_constraint_penalty,
            'density_bonus': weighted_density_bonus,
            'total_elbo': total_elbo
        }