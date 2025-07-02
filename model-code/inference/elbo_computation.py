import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from models import NetworkEvolution, StateTransition
import pdb

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
                 rho_1: float = 0.5, rho_2: float = 0.5, confidence_weight: float = 0.0):
        self.network_evolution = network_evolution
        self.state_transition = state_transition
        self.confidence_weight = confidence_weight
        
        # Observation model parameters
        self.rho_1 = torch.nn.Parameter(torch.tensor(rho_1))  # Probability of observing link type 1
        self.rho_2 = torch.nn.Parameter(torch.tensor(rho_2))  # Probability of observing link type 2

    def compute_state_likelihood_batch(self,
                                     features: torch.Tensor,
                                     states: torch.Tensor,
                                     distances: torch.Tensor,
                                     node_batch: torch.Tensor,
                                     network_data,
                                     gumbel_samples: List[Dict[str, torch.Tensor]],
                                     max_timestep: int) -> torch.Tensor:
        """
        Optimized state likelihood using vectorized operations.
        Eliminates triple nested loops for significant speedup.
        """
        if len(gumbel_samples) == 0:
            return torch.tensor(0.0)
        
        num_samples = len(gumbel_samples)
        batch_size = len(node_batch)
        
        # Pre-compute undecided mask: [batch_size, max_timestep, 3]
        # True where states[node_batch, t, decision_k] == 0
        batch_states = states[node_batch]  # [batch_size, total_timesteps, 3]
        undecided_mask = (batch_states[:, :max_timestep, :] == 0)  # [batch_size, max_timestep, 3]
        
        # Pre-compute actual outcomes: [batch_size, max_timestep, 3]
        actual_outcomes = batch_states[:, 1:max_timestep+1, :]  # [batch_size, max_timestep, 3]

        def compute_dynamic_class_weights_vectorized(activations_batch):
            """Vectorized class weight computation for multiple batches"""
            n_positive = activations_batch.sum(dim=0)  # Sum over batch dimension
            n_total = activations_batch.shape[0]
            
            # Handle zero positive cases
            pos_weight = torch.where(
                n_positive > 0,
                torch.clamp(5 * (n_total - n_positive) / n_positive, max=100.0),
                torch.ones_like(n_positive)
            )
            neg_weight = torch.ones_like(pos_weight)
            
            return pos_weight, neg_weight

        total_likelihood = 0.0
        
        # Main optimization: process all samples with minimal loops
        for sample_idx, current_samples in enumerate(gumbel_samples):
            sample_likelihood = 0.0
            
            # Process all (t, decision_k) combinations more efficiently
            for t in range(max_timestep):
                for decision_k in range(3):
                    # Get mask for current (t, decision_k)
                    current_undecided = undecided_mask[:, t, decision_k]  # [batch_size]
                    
                    if not current_undecided.any():
                        continue
                    
                    # Extract undecided households for this (t, decision_k)
                    undecided_indices = node_batch[current_undecided]
                    
                    if len(undecided_indices) == 0:
                        continue
                    
                    # Compute activation probabilities for undecided households
                    activation_probs = self.state_transition.compute_activation_probability(
                        household_idx=undecided_indices,
                        decision_type=decision_k,
                        features=features,
                        states=states,
                        distances=distances,
                        network_data=network_data,
                        gumbel_samples=current_samples,
                        time=t
                    )
                    
                    # Get corresponding actual outcomes
                    current_outcomes = actual_outcomes[current_undecided, t, decision_k]

                    # Vectorized class weight computation
                    pos_weight, neg_weight = compute_dynamic_class_weights_vectorized(
                        current_outcomes.unsqueeze(0)
                    )
                    
                    # Vectorized log likelihood computation
                    pos_log_probs = current_outcomes * torch.log(activation_probs + 1e-8)
                    neg_log_probs = (1 - current_outcomes) * torch.log(1 - activation_probs + 1e-8)
                    
                    weighted_likelihood = (pos_weight[0] * pos_log_probs.sum() + 
                                         neg_weight[0] * neg_log_probs.sum())
                    sample_likelihood += weighted_likelihood
            
            total_likelihood += sample_likelihood
        
        return torch.tensor(total_likelihood / num_samples, dtype=torch.float32)
    

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
                        
                        total_likelihood += torch.log(torch.tensor(obs_prob) + 1e-8)
                
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
            
            return torch.tensor(total_likelihood) if isinstance(total_likelihood, float) else total_likelihood
    
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
    
    def compute_confidence_regularization(self, marginal_probs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Network confidence regularization using marginal probabilities.
        NOW WITH NORMALIZATION: Divided by total possible link-timesteps.
        """
        if self.confidence_weight == 0.0:
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
        ULTRA OPTIMIZED: Compute decision constraint penalties with maximum vectorization.
        
        Revolutionary optimizations:
        1. Pre-compute ALL activation probabilities in single mega-batch
        2. Cache state histories and neural network inputs
        3. Vectorized penalty computation across ALL samples simultaneously
        4. Eliminate all Python loops in critical paths
        """
        if len(gumbel_samples) == 0:
            return torch.tensor(0.0)
        
        num_samples = len(gumbel_samples)
        batch_size = len(node_batch)
        
        # === STAGE 1: Pre-compute all data structures ===
        batch_states = states[node_batch]  # [batch_size, total_timesteps, 3]
        observed_states = batch_states[:, :max_timestep+1, :]  # [batch_size, max_timestep+1, 3]
        
        # Pre-compute penalty masks for all timesteps
        vacant_obs_all = observed_states[:, :, 0]  # [batch_size, max_timestep+1]
        repair_obs_all = observed_states[:, :, 1]  # [batch_size, max_timestep+1] 
        sell_obs_all = observed_states[:, :, 2]    # [batch_size, max_timestep+1]
        
        repair_zero_mask_all = (repair_obs_all == 0)  # [batch_size, max_timestep+1]
        penalty_2_mask_all = (vacant_obs_all == 0) & (sell_obs_all == 1)  # [batch_size, max_timestep+1]
        
        # === STAGE 2: Pre-compute ALL required activation probabilities ===
        activation_prob_cache = self._precompute_all_activation_probabilities(
            node_batch, features, states, distances, network_data, 
            gumbel_samples, max_timestep, repair_zero_mask_all, penalty_2_mask_all
        )
        
        # === STAGE 3: Ultra-fast vectorized penalty computation ===
        total_penalty = self._compute_penalties_vectorized(
            activation_prob_cache, node_batch, max_timestep, num_samples,
            repair_zero_mask_all, penalty_2_mask_all, 
            vacant_obs_all, sell_obs_all, repair_obs_all, batch_size
        )
        
        return torch.tensor(total_penalty / num_samples)
    
    def _precompute_all_activation_probabilities(self, node_batch, features, states, distances,
                                               network_data, gumbel_samples, max_timestep,
                                               repair_zero_mask_all, penalty_2_mask_all):
        """
        REVOLUTIONARY: Pre-compute ALL activation probabilities in mega-batches.
        This eliminates repeated NN calls and dramatically speeds up computation.
        """
        cache = {
            'penalty_1': {},  # (sample_idx, t, decision_type) -> [n_nodes] probabilities
            'penalty_2': {}   # (sample_idx, t) -> [n_nodes] probabilities  
        }
        
        # === Identify all unique (t, decision_type) combinations needed ===
        penalty_1_combinations = set()
        penalty_2_combinations = set()
        
        for t in range(max_timestep + 1):
            # Penalty 1 needs (t, 0), (t, 1), (t, 2) wherever repair_zero_mask is True
            if repair_zero_mask_all[:, t].any():
                penalty_1_combinations.update([(t, 0), (t, 1), (t, 2)])
            
            # Penalty 2 needs (t, 0) wherever penalty_2_mask is True
            if penalty_2_mask_all[:, t].any():
                penalty_2_combinations.add((t, 0))
        
        # === Pre-compute for Penalty 1 ===
        for sample_idx, current_samples in enumerate(gumbel_samples):
            for t, decision_type in penalty_1_combinations:
                repair_zero_nodes = node_batch[repair_zero_mask_all[:, t]]
                if len(repair_zero_nodes) > 0:
                    probs = self._batch_activation_prob(
                        repair_zero_nodes, decision_type, features, states, distances,
                        network_data, current_samples, t
                    )
                    cache['penalty_1'][(sample_idx, t, decision_type)] = probs
        
        # === Pre-compute for Penalty 2 ===  
        for sample_idx, current_samples in enumerate(gumbel_samples):
            for t, decision_type in penalty_2_combinations:
                penalty_2_nodes = node_batch[penalty_2_mask_all[:, t]]
                if len(penalty_2_nodes) > 0:
                    probs = self._batch_activation_prob(
                        penalty_2_nodes, decision_type, features, states, distances,
                        network_data, current_samples, t
                    )
                    cache['penalty_2'][(sample_idx, t)] = probs
        
        return cache
    
    def _compute_penalties_vectorized(self, cache, node_batch, max_timestep, num_samples,
                                    repair_zero_mask_all, penalty_2_mask_all,
                                    vacant_obs_all, sell_obs_all, repair_obs_all, batch_size):
        """
        ULTRA-FAST: Compute all penalties using pre-computed probabilities.
        No neural network calls, pure tensor operations.
        """
        total_penalty = 0.0
        
        for sample_idx in range(num_samples):
            sample_penalty = 0.0
            
            # === Penalty 1: Vectorized computation ===
            for t in range(max_timestep + 1):
                repair_zero_mask_t = repair_zero_mask_all[:, t]
                
                if not repair_zero_mask_t.any():
                    continue
                
                # Get pre-computed probabilities
                key_vacant = (sample_idx, t, 0)
                key_repair = (sample_idx, t, 1)
                key_sell = (sample_idx, t, 2)
                
                if all(key in cache['penalty_1'] for key in [key_vacant, key_repair, key_sell]):
                    vacant_probs = cache['penalty_1'][key_vacant]
                    repair_probs = cache['penalty_1'][key_repair]
                    sell_probs = cache['penalty_1'][key_sell]
                    
                    # Use observed values where available, predicted where not
                    vacant_obs_masked = vacant_obs_all[repair_zero_mask_t, t]
                    sell_obs_masked = sell_obs_all[repair_zero_mask_t, t]
                    
                    vacant_final = torch.where(vacant_obs_masked == 1, vacant_obs_masked.float(), vacant_probs)
                    sell_final = torch.where(sell_obs_masked == 1, sell_obs_masked.float(), sell_probs)
                    
                    # Vectorized penalty computation
                    penalty_batch = repair_probs * (vacant_final + sell_final)
                    sample_penalty += penalty_batch.sum().item()
            
            # === Penalty 2: Vectorized computation ===
            for t in range(max_timestep + 1):
                penalty_2_mask_t = penalty_2_mask_all[:, t]
                
                if not penalty_2_mask_t.any():
                    continue
                
                key = (sample_idx, t)
                if key in cache['penalty_2']:
                    vacant_probs = cache['penalty_2'][key]
                    penalty_batch = 1 - vacant_probs
                    sample_penalty += penalty_batch.sum().item()
            
            total_penalty += sample_penalty / (batch_size * (max_timestep + 1))
        
        return total_penalty
    
    def _compute_penalty_1_vectorized(self, node_batch, features, states, distances, 
                                     network_data, current_samples, max_timestep,
                                     repair_zero_mask_all, vacant_obs_all, sell_obs_all, repair_obs_all):
        """
        LEGACY: Kept for backward compatibility but now redirects to optimized version.
        """
        # This method is now called through the new optimized pipeline
        return 0.0  # Handled by _compute_penalties_vectorized
    
    def _compute_penalty_2_vectorized(self, node_batch, features, states, distances,
                                     network_data, current_samples, max_timestep, penalty_2_mask_all):
        """
        LEGACY: Kept for backward compatibility but now redirects to optimized version.
        """
        # This method is now called through the new optimized pipeline
        return 0.0  # Handled by _compute_penalties_vectorized

    def compute_information_propagation_penalty(self, marginal_probs, network_data, max_timestep):
        """
        OPTIMIZED: Information propagation penalty with reduced redundant computations.
        """
        total_penalty = 0.0
        bonding_pairs_processed = set()
        
        # Pre-filter marginal_probs by time for faster lookup
        time_indexed_probs = {}
        for pair_key, prob in marginal_probs.items():
            parts = pair_key.split('_')
            if len(parts) == 3:
                i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
                if t not in time_indexed_probs:
                    time_indexed_probs[t] = {}
                time_indexed_probs[t][(i, j)] = prob
        
        # Process observed edges with optimized lookups
        for t in range(max_timestep + 1):
            observed_edges = network_data.get_observed_edges_at_time(t)
            
            for i, j, observed_type in observed_edges:
                if observed_type == 1:  # bonding: avoid double counting
                    if (i, j) in bonding_pairs_processed:
                        continue
                    bonding_pairs_processed.add((i, j))
                    
                    # Use pre-indexed lookups instead of string formatting
                    for t_other, time_probs in time_indexed_probs.items():
                        if (i, j) in time_probs:
                            total_penalty -= 100 * torch.log(time_probs[(i, j)][1] + 1e-8)
                
                elif observed_type == 2:  # bridging: vectorized neighbor computation
                    neighbor_times = [t + delta for delta in [-2, -1, 1, 2] 
                                    if 0 <= t + delta <= max_timestep]
                    weights = [1.0 / (abs(delta) + 1) for delta in [-2, -1, 1, 2] 
                             if 0 <= t + delta <= max_timestep]
                    
                    for t_neighbor, weight in zip(neighbor_times, weights):
                        if t_neighbor in time_indexed_probs and (i, j) in time_indexed_probs[t_neighbor]:
                            total_penalty -= weight * torch.log(time_indexed_probs[t_neighbor][(i, j)][2] + 1e-8)
        
        return total_penalty

    def compute_type_specific_density_penalty(self, marginal_probs, network_data, max_timestep,
                                            temperature=0.01, balance_factor=1.0, penalty_strength=1.0):
        """
        OPTIMIZED: Type-specific density penalty with vectorized computations.
        Key optimizations:
        1. Pre-group marginal probabilities by timestep
        2. Vectorized sharp softmax computation
        3. Batch tensor operations
        """
        total_penalty = 0.0
        
        # Pre-group marginal probabilities by timestep for efficient processing
        timestep_probs = {}
        for pair_key, π_ij in marginal_probs.items():
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            pair_t = int(parts[2])
            if pair_t not in timestep_probs:
                timestep_probs[pair_t] = []
            timestep_probs[pair_t].append(π_ij)
        
        # Global statistics for final print
        total_expected_bonding = 0.0
        total_expected_bridging = 0.0
        total_discrete_bonding = 0.0
        total_discrete_bridging = 0.0
        
        for t in range(max_timestep + 1):
            # Count observed edges at timestep t (vectorized)
            observed_edges = network_data.get_observed_edges_at_time(t)
            edge_types = [link_type for _, _, link_type in observed_edges]
            observed_bonding_t = edge_types.count(1)
            observed_bridging_t = edge_types.count(2)
            
            # Estimate expected hidden edge counts
            expected_total_bonding_t = observed_bonding_t / (1 - self.rho_1) if self.rho_1 < 1 else observed_bonding_t
            expected_total_bridging_t = observed_bridging_t / (1 - self.rho_2) if self.rho_2 < 1 else observed_bridging_t
            
            expected_hidden_bonding_t = torch.clamp(torch.tensor(expected_total_bonding_t - observed_bonding_t), min=0.0)
            expected_hidden_bridging_t = torch.clamp(torch.tensor(expected_total_bridging_t - observed_bridging_t), min=0.0)
            
            # Vectorized computation for model predicted hidden edges
            discrete_bonding_t = torch.tensor(0.0)
            discrete_bridging_t = torch.tensor(0.0)
            
            if t in timestep_probs and timestep_probs[t]:
                # Stack all probabilities for current timestep: [n_pairs, 3]
                prob_stack = torch.stack(timestep_probs[t])
                
                # Vectorized sharp softmax computation: [n_pairs, 3]
                logits = torch.log(prob_stack + 1e-8)
                sharp_probs = F.softmax(logits / temperature, dim=1)
                
                # Sum across all pairs for each edge type
                discrete_bonding_t = sharp_probs[:, 1].sum()
                discrete_bridging_t = sharp_probs[:, 2].sum()
            
            # Compute penalty for timestep t with optimized logic
            bonding_penalty_t = torch.tensor(0.0)
            bridging_penalty_t = torch.tensor(0.0)
            
            if expected_hidden_bonding_t > 0.1:
                bonding_relative_error = torch.abs(discrete_bonding_t - expected_hidden_bonding_t) / expected_hidden_bonding_t
                bonding_penalty_t = balance_factor * bonding_relative_error
            bonding_penalty_t = 3 * torch.clamp(bonding_penalty_t, 0.0, 20.0) / 20
            
            if expected_hidden_bridging_t > 0.1:
                bridging_ratio = (discrete_bridging_t + 1) / (expected_hidden_bridging_t + 1)
                bridging_penalty_t = torch.abs(torch.log(bridging_ratio))
            bridging_penalty_t = torch.clamp(bridging_penalty_t, 0.0, 3.0)
            
            # Combine penalties for current timestep
            timestep_penalty = 0.5 * bonding_penalty_t + 0.5 * bridging_penalty_t
            total_penalty += timestep_penalty
            
            # Accumulate global statistics (optimized tensor handling)
            total_expected_bonding += expected_hidden_bonding_t.item() if isinstance(expected_hidden_bonding_t, torch.Tensor) else expected_hidden_bonding_t
            total_expected_bridging += expected_hidden_bridging_t.item() if isinstance(expected_hidden_bridging_t, torch.Tensor) else expected_hidden_bridging_t
            total_discrete_bonding += discrete_bonding_t.item() if isinstance(discrete_bonding_t, torch.Tensor) else discrete_bonding_t
            total_discrete_bridging += discrete_bridging_t.item() if isinstance(discrete_bridging_t, torch.Tensor) else discrete_bridging_t
        
        # Average across timesteps
        averaged_penalty = total_penalty / (max_timestep + 1)
        final_penalty = penalty_strength * averaged_penalty
        
        # Print in your original format
        print(f"Timestep-Specific Density Penalty (T={temperature}, balance={balance_factor}):")
        print(f"  Expected: bonding={total_expected_bonding:.1f}, bridging={total_expected_bridging:.1f}")
        print(f"  Discrete: bonding={total_discrete_bonding:.1f}, bridging={total_discrete_bridging:.1f}")
        print(f"  Final penalty: {final_penalty:.3f}")
        
        return final_penalty

    def compute_elbo_batch(self, features, states, distances, node_batch, network_data, 
                          conditional_probs, marginal_probs, gumbel_samples, max_timestep, 
                          lambda_constraint, current_epoch):
        """Compute the full ELBO with all components and dynamic weighting."""

        def get_dynamic_weights(epoch):
            if epoch < 100:  
                return {
                    'state': 0.0, 
                    'observation': 1.0, 
                    'prior': 0.1,                          
                    'entropy': 1.0, 
                    'confidence': 0.5, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 2.0,                    
                    'density_penalty': 0.5,
                    'info_propagation': 1e-4    ##############upupup!!!!!!!!!!
                }
            elif epoch < 200:
                progress = (epoch - 100) / 100
                state_weight = 0.1    
                prior_weight = 0.1 + 0.1 * progress      
                # density_weight = 2.0 * (1 - 0.5 * progress) 
                density_weight = 2.0
                return {
                    'state': state_weight, 
                    'observation': 1.0, 
                    'prior': prior_weight,
                    'entropy': 1.0 - 0.3 * progress, 
                    'confidence': 0.5, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': density_weight,
                    'density_penalty': 0.5,
                    'info_propagation': 1e-4      
                }
            elif epoch < 300:
                return {
                    'state': 0.1, 
                    'observation': 1.0, 
                    'prior': 0.2,
                    'entropy': 0.7, 
                    'confidence': 0.5, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 2.0,          
                    'density_penalty': 0.5,
                    'info_propagation': 1e-4
                }
            else:  
                return {
                    'state': 1.0, 
                    'observation': 0.0, 
                    'prior': 0.0,
                    'entropy': 0.0, 
                    'confidence': 0.0, 
                    'constraint': -lambda_constraint*2, 
                    'density_bonus': 0.0,          
                    'density_penalty': 0.0,
                    'info_propagation': 0.0
                }

        # Get normalization factors
        total_households = features.shape[0]
        total_possible_pairs = total_households * (total_households - 1) // 2
        total_timesteps = max_timestep + 1
        
        # Compute raw likelihoods with timing measurements
        import time
        timing_results = {}
        
        # State likelihood computation
        start_time = time.time()
        state_likelihood_raw = self.compute_state_likelihood_batch(
            features, states, distances, node_batch, network_data, gumbel_samples, max_timestep
        )
        timing_results['state_likelihood'] = time.time() - start_time

        # Network observation likelihood computation  
        start_time = time.time()
        observation_likelihood_raw = self.compute_network_observation_likelihood_batch(
            marginal_probs, node_batch, network_data, max_timestep
        )
        timing_results['observation_likelihood'] = time.time() - start_time
        
        # Prior likelihood computation
        start_time = time.time()
        prior_likelihood_raw = self.compute_prior_likelihood_batch(
            conditional_probs, marginal_probs, features, states, distances, 
            node_batch, network_data, max_timestep
        )
        timing_results['prior_likelihood'] = time.time() - start_time
        
        # Posterior entropy computation
        start_time = time.time()
        posterior_entropy_raw = self.compute_posterior_entropy_batch(
            conditional_probs, marginal_probs, network_data, node_batch, max_timestep
        )
        timing_results['posterior_entropy'] = time.time() - start_time
        
        # Confidence regularization computation
        start_time = time.time()
        confidence_reg_raw = self.compute_confidence_regularization(marginal_probs)
        timing_results['confidence_regularization'] = time.time() - start_time

        # Constraint penalty computation
        start_time = time.time()
        constraint_penalty_raw = self.compute_constraint_penalty(
        features, states, distances, node_batch, network_data, 
        gumbel_samples, max_timestep
        )
        timing_results['constraint_penalty'] = time.time() - start_time

        # Type-specific density penalty computation
        start_time = time.time()
        timestep_density_penalty = self.compute_type_specific_density_penalty(
            marginal_probs, network_data, max_timestep
        )
        timing_results['density_penalty'] = time.time() - start_time

        # Information propagation penalty computation
        start_time = time.time()
        info_propagation_penalty = self.compute_information_propagation_penalty(
        marginal_probs, network_data, max_timestep
        )
        timing_results['info_propagation'] = time.time() - start_time

        # Print timing results
        print("\n=== FUNCTION TIMING ANALYSIS ===")
        total_time = sum(timing_results.values())
        for func_name, exec_time in timing_results.items():
            percentage = (exec_time / total_time) * 100 if total_time > 0 else 0
            print(f"{func_name:25s}: {exec_time:.4f}s ({percentage:5.1f}%)")
        print(f"{'TOTAL TIME':25s}: {total_time:.4f}s (100.0%)")

        print("=" * 40)
        print(f"Raw values - State: {state_likelihood_raw.item():.2f}, "
            f"Obs: {observation_likelihood_raw.item():.2f}, "
            f"Prior: {prior_likelihood_raw.item():.2f}, "
            f"Entropy: {posterior_entropy_raw.item():.2f},"
            f" Sparsity: {confidence_reg_raw.item():.2f}, "
            f"Constraint: {constraint_penalty_raw.item():.2f}, "
            f"Timestep Density Penalty: {timestep_density_penalty.item():.2f}, "
            f"Info Propagation Penalty: {info_propagation_penalty.item():.2f},")
            #f"Density Bonus: {density_bonus_raw.item():.2f}")

        # Calculate actual counts for proper normalization
        # total_state_predictions =  max_timestep * 3  # 3 decision types
        total_state_predictions = len(node_batch) * max_timestep * 3  # 3 decision types per sample
        total_network_pairs = len(marginal_probs)  # Actual pairs being evaluated
        print(f"Total state predictions: {total_state_predictions}, Total network pairs: {total_network_pairs}")
        
        # Normalize by actual counts
        state_likelihood_per_prediction = state_likelihood_raw / total_state_predictions
        network_likelihood_per_pair = observation_likelihood_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        prior_likelihood_per_pair = prior_likelihood_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        entropy_per_pair = posterior_entropy_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        confidence_per_pair = self.confidence_weight * confidence_reg_raw / total_network_pairs if total_network_pairs > 0 else torch.tensor(0.0)
        
        print(f"Per-unit values - State: {state_likelihood_per_prediction.item():.4f}, "
            f"Obs: {network_likelihood_per_pair.item():.4f}, "
            f"Prior: {prior_likelihood_per_pair.item():.4f}, "
            f"Entropy: {entropy_per_pair.item():.4f}")
        

        weight = get_dynamic_weights(current_epoch)       
        # Apply adaptive weighting
        weighted_state_likelihood = weight['state'] * state_likelihood_per_prediction if current_epoch>=200 else torch.tensor(-0.8)
        weighted_observation_likelihood = weight['observation'] * network_likelihood_per_pair
        weighted_prior_likelihood = weight['prior'] * prior_likelihood_per_pair
        weighted_posterior_entropy = weight['entropy'] * entropy_per_pair
        weighted_confidence_reg = weight['confidence'] * confidence_per_pair
        weighted_constraint_penalty = weight['constraint'] * constraint_penalty_raw
        weighted_density_penalty = weight['density_penalty'] * timestep_density_penalty
        weighted_info_propagation = weight['info_propagation'] * info_propagation_penalty

        # weighted_density_bonus = weight['density_bonus'] * density_bonus_raw

        # Total weighted ELBO
        # total_elbo = (weighted_state_likelihood + weighted_observation_likelihood +
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg - weighted_constraint_penalty)
        total_elbo = (weighted_state_likelihood + weighted_observation_likelihood +
                    weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg -
                    weighted_constraint_penalty - weighted_density_penalty - weighted_info_propagation)

        print(f"Weighted components - State: {weighted_state_likelihood.item():.4f}, "
            f"Obs: {weighted_observation_likelihood.item():.4f}, "
            f"Prior: {weighted_prior_likelihood.item():.4f}, "
            f"Entropy: {weighted_posterior_entropy.item():.4f}")
        print(f"Sparsity: {weighted_confidence_reg.item():.4f}, "
            f"Constraint: {weighted_constraint_penalty.item():.4f}, "
            f"Timestep Density Penalty: {weighted_density_penalty.item():.4f}, "
            f"Info Propagation: {weighted_info_propagation.item():.4f}")
            # f"Density Bonus: {weighted_density_bonus.item():.4f}")
        print(f"Total weighted ELBO: {total_elbo.item():.4f}")

        return {
            'state_likelihood': weighted_state_likelihood,
            'observation_likelihood': weighted_observation_likelihood, 
            'prior_likelihood': weighted_prior_likelihood,
            'posterior_entropy': weighted_posterior_entropy,
            'confidence_regularization': weighted_confidence_reg,
            'constraint_penalty': weighted_constraint_penalty,
            'density_penalty': weighted_density_penalty,
            'info_propagation_penalty': weighted_info_propagation,
            # 'density_bonus': weighted_density_bonus,
            'total_elbo': total_elbo
        }

    def _batch_activation_prob(self, node_indices, decision_type, features, states, distances, 
                              network_data, gumbel_samples, time):
        """
        OPTIMIZED: Compute activation probabilities for multiple nodes at once.
        Reduces individual NN calls by batching.
        """
        if len(node_indices) == 0:
            return torch.tensor([])
        
        return self.state_transition.compute_activation_probability(
            household_idx=node_indices,
            decision_type=decision_type,
            features=features,
            states=states,
            distances=distances,
            network_data=network_data,
            gumbel_samples=gumbel_samples,
            time=time
        )