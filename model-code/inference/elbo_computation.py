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
                 rho_1: float = 0.5, rho_2: float = 0.5, confidence_weight: float = 0.0,):
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
                                    max_timestep: int,
                                    current_epoch: int = 0) -> torch.Tensor:
        """
        State likelihood using marginal-based Gumbel-Softmax samples.
        NOW WITH NORMALIZATION: Divided by total possible decision opportunities.
        """

        def compute_dynamic_class_weights(current_activations, current_epoch):
            n_positive = current_activations.sum()
            n_total = len(current_activations)
            
            if n_positive == 0:
                return 1.0, 1.0  
            
            # Base positive weight calculation
            base_pos_weight = min((n_total - n_positive) / n_positive, 5.0)
            
            # Adjust based on training phase
            if current_epoch < 300:  # Network learning phase
                pos_weight = base_pos_weight  # Keep current penalty
            elif current_epoch < 350:  # Bridging phase
                progress = (current_epoch - 300) / 50
                pos_weight = base_pos_weight * (1 - 0.4 * progress)  # Reduce by 40%
            else:  # Rollout-dominated phase
                pos_weight = base_pos_weight * 0.6  # Reduce to 60%
            
            return pos_weight, 1.0  # Negative weight fixed at 1.0

        total_likelihood = 0.0
        num_samples = len(gumbel_samples)
        # print(f"Number of Gumbel samples: {num_samples}")
        
        for sample_idx, current_samples in enumerate(gumbel_samples):
            self.state_transition.broken_links_history.clear()
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

                    pos_weight, neg_weight = compute_dynamic_class_weights(actual_outcomes, current_epoch)
                    
                    # Compute log likelihood
                    pos_log_probs = actual_outcomes * torch.log(activation_probs + 1e-8)
                    neg_log_probs = (1 - actual_outcomes) * torch.log(1 - activation_probs + 1e-8)
                    
                    weighted_likelihood = (pos_weight * pos_log_probs.sum() + 
                                        neg_weight * neg_log_probs.sum())
                    sample_likelihood += weighted_likelihood
            
            total_likelihood += sample_likelihood
        
        return total_likelihood/num_samples
    

    def compute_rollout_state_likelihood_batch(self,
                                          features: torch.Tensor,
                                          states: torch.Tensor,
                                          distances: torch.Tensor,
                                          node_batch: torch.Tensor,
                                          network_data,
                                          gumbel_samples: List[Dict[str, torch.Tensor]],
                                          max_timestep: int,
                                          current_epoch: int,
                                          rollout_steps: int = 3,
                                          use_pred_prob: float = 0.3) -> torch.Tensor:
        """
        Rollout training: multi-step forward with scheduled sampling
        
        Args:
            rollout_steps: Number of forward steps to roll out
            use_pred_prob: Probability of using model prediction vs ground truth
        """
        if not gumbel_samples or rollout_steps <= 0:
            return torch.tensor(0.0, device=features.device)

        device = features.device
        total_likelihood = 0.0
        num_samples = len(gumbel_samples)
        
        print(f"Starting rollout training: {rollout_steps} steps, use_pred_prob={use_pred_prob:.2f}")

        for sample_idx, current_samples in enumerate(gumbel_samples):
            self.state_transition.broken_links_history.clear()
            sample_likelihood = 0.0
            
            # Select rollout starting time points
            max_start_time = max_timestep - rollout_steps
            if max_start_time <= 0:
                continue
                
            for start_t in range(0, max_start_time, rollout_steps):  # Start every rollout_steps
                # Initialize: use ground truth states as starting point
                current_rollout_states = states[:, start_t, :].clone()  # [N, 3]
                
                step_likelihood = 0.0
                
                for step in range(rollout_steps):
                    actual_t = start_t + step
                    if actual_t >= max_timestep:
                        break
                    
                    for decision_k in range(3):
                        # Find inactive households
                        batch_undecided = []
                        for node_idx in node_batch:
                            if current_rollout_states[node_idx, decision_k] == 0:
                                batch_undecided.append(node_idx)
                        
                        if len(batch_undecided) == 0:
                            continue
                        
                        batch_undecided_tensor = torch.tensor(batch_undecided, dtype=torch.long)
                        
                        # Build temporary state tensor for prediction
                        temp_states = states.clone()
                        temp_states[:, actual_t, :] = current_rollout_states
                        
                        # Compute activation probabilities
                        activation_probs = self.state_transition.compute_activation_probability(
                            household_idx=batch_undecided_tensor,
                            decision_type=decision_k,
                            features=features,
                            states=temp_states,
                            distances=distances,
                            network_data=network_data,
                            gumbel_samples=current_samples,
                            time=actual_t
                        )
                        
                        # Get ground truth labels
                        true_outcomes = states[batch_undecided, actual_t + 1, decision_k]
                        
                        # Dynamic class weights with epoch-based adjustment
                        n_positive = true_outcomes.sum()
                        n_total = len(true_outcomes)
                        if n_positive > 0:
                            base_pos_weight = min((n_total - n_positive) / n_positive, 5.0)
                            
                            # Adjust penalty based on current epoch (same logic as regular training)
                            if current_epoch < 300:  # Network learning phase
                                pos_weight = base_pos_weight  # Keep current penalty
                            elif current_epoch < 350:  # Bridging phase
                                progress = (current_epoch - 300) / 50
                                pos_weight = base_pos_weight * (1 - 0.4 * progress)  # Reduce by 40%
                            else:  # Rollout-dominated phase
                                pos_weight = base_pos_weight * 0.6  # Reduce to 60%
                        else:
                            pos_weight = 1.0
                        
                        # Compute loss
                        pos_log_probs = true_outcomes * torch.log(activation_probs + 1e-8)
                        neg_log_probs = (1 - true_outcomes) * torch.log(1 - activation_probs + 1e-8)
                        weighted_likelihood = (pos_weight * pos_log_probs.sum() + 
                                            1.0 * neg_log_probs.sum())
                        step_likelihood += weighted_likelihood
                        
                        # Scheduled Sampling: update rollout states
                        if step < rollout_steps - 1:  # Don't update on the last step
                            for idx, node_idx in enumerate(batch_undecided):
                                if torch.rand(1) < use_pred_prob:
                                    # Use model prediction
                                    predicted = (activation_probs[idx] > 0.5).float()
                                    current_rollout_states[node_idx, decision_k] = predicted
                                else:
                                    # Use ground truth
                                    current_rollout_states[node_idx, decision_k] = true_outcomes[idx]
                
                sample_likelihood += step_likelihood
            
            total_likelihood += sample_likelihood

        rollout_avg = total_likelihood / num_samples
        print(f"Rollout likelihood: {rollout_avg.item():.4f}")
        return rollout_avg
      

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
            
            # Get ALL distance-filtered pairs involving batch nodes (not just hidden!)
            def get_all_batch_pairs(t):
                all_pairs = []
                batch_nodes_set = set(node_batch.tolist())
                neighbor_index = getattr(network_data, "neighbor_index", None)
                N = features.shape[0]

                if neighbor_index is None:
                    # old behavior
                    for i in range(N):
                        for j in range(i + 1, N):
                            if i in batch_nodes_set or j in batch_nodes_set:
                                all_pairs.append((i, j))
                    return all_pairs

                # sparse behavior: only i < j and neighbor-filtered
                for i in range(N):
                    for j in neighbor_index[i]:
                        if j <= i:
                            continue
                        if (i in batch_nodes_set) or (j in batch_nodes_set):
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
    #     """Reward network connectivity to counteract confidence bias"""
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

    # def compute_connection_density_bonus(self, marginal_probs, target_density=0.35):
    #     """Target-based density bonus instead of always rewarding connections"""
        
    #     connection_probs = []
    #     for pair_key, π_ij in marginal_probs.items():
    #         connection_prob = π_ij[1] + π_ij[2]
    #         connection_probs.append(connection_prob)
        
    #     if len(connection_probs) > 0:
    #         current_density = torch.mean(torch.stack(connection_probs))
            
    #         # If current density is below target, reward based on log difference
    #         if current_density < target_density:
    #             bonus = torch.log(current_density / target_density + 1e-8)  
    #         else:
    #             bonus = - (current_density - target_density) ** 2      
            
    #         return bonus
        
    #     return torch.tensor(0.0)
    
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
                                            temperature=0.01, balance_factor=1.0,
                                            penalty_strength=1.0,
                                            rho1_scale=1.0, rho2_scale=1.0):
        """
        Sparse-aware density penalty:
        - Only uses candidate (top-k/radius) pairs for both observed counts and predictions.
        - Expected hidden inside candidates = m * (rho / (1 - rho)).
        - Optionally shrink missing rates inside candidates via rho*_scale in (0,1] if desired.

        If network_data.neighbor_index is None, this reduces to your previous dense behavior
        because all pairs are treated as candidates.
        """
        import torch
        import torch.nn.functional as F

        device = next((v.device for v in marginal_probs.values()), torch.device("cpu")) \
                if len(marginal_probs) > 0 else torch.device("cpu")
        eps = 1e-8

        neighbor_index = getattr(network_data, "neighbor_index", None)
        def in_cand(i, j):
            if neighbor_index is None:
                return True
            return (j in neighbor_index[i]) or (i in neighbor_index[j])

        # group candidate marginals by timestep (same as before)
        timestep_probs = {}
        for key, pij in marginal_probs.items():
            parts = key.split('_')
            if len(parts) != 3: continue
            t = int(parts[2])
            timestep_probs.setdefault(t, []).append(pij)

        total_penalty = torch.tensor(0.0, device=device)

        # logs (for visibility)
        tot_exp_hidden_bond = 0.0
        tot_exp_hidden_brid = 0.0
        tot_pred_bond = 0.0
        tot_pred_brid = 0.0
        tot_cand_pairs = 0

        # candidate-effective missing rates (optionally scaled)
        r1 = max(0.0, min(0.999, float(self.rho_1) * float(rho1_scale)))
        r2 = max(0.0, min(0.999, float(self.rho_2) * float(rho2_scale)))
        r1_tensor = torch.tensor(r1, device=device)
        r2_tensor = torch.tensor(r2, device=device)

        for t in range(max_timestep + 1):
            # 1) count observed positives in candidates at t
            m1 = 0  # observed bonding inside candidates
            m2 = 0  # observed bridging inside candidates
            observed_edges_t = network_data.get_observed_edges_at_time(t)  # list of (i,j,link_type)
            for (i, j, link_type) in observed_edges_t:
                if in_cand(i, j):
                    if link_type == 1: m1 += 1
                    elif link_type == 2: m2 += 1

            m1_t = torch.tensor(float(m1), device=device)
            m2_t = torch.tensor(float(m2), device=device)

            # 2) expected hidden in candidates via simplified formula m * rho/(1-rho)
            exp_hidden_bond_t = m1_t * (r1_tensor / (1.0 - r1_tensor + eps))
            exp_hidden_brid_t = m2_t * (r2_tensor / (1.0 - r2_tensor + eps))

            # 3) model predicted counts on candidates at t (your sharp-softmax trick)
            pred_bond_t = torch.tensor(0.0, device=device)
            pred_brid_t = torch.tensor(0.0, device=device)
            n_cand_t = 0
            if t in timestep_probs and len(timestep_probs[t]) > 0:
                prob_stack = torch.stack(timestep_probs[t]).to(device)   # [P,3], P=candidate pairs at t
                n_cand_t = prob_stack.shape[0]
                logits = torch.log(prob_stack + eps)
                sharp = F.softmax(logits / temperature, dim=1)
                pred_bond_t = sharp[:, 1].sum()
                pred_brid_t = sharp[:, 2].sum()

            # 4) penalties
            bond_pen_t = torch.tensor(0.0, device=device)
            if exp_hidden_bond_t.item() > 0.1:
                rel_err = torch.abs(pred_bond_t - exp_hidden_bond_t) / (exp_hidden_bond_t + eps)
                bond_pen_t = balance_factor * rel_err
            bond_pen_t = 5.0 * torch.clamp(bond_pen_t, 0.0, 20.0) / 20.0

            brid_pen_t = torch.tensor(0.0, device=device)
            if exp_hidden_brid_t.item() > 0.1:
                ratio = (pred_brid_t + 1.0) / (exp_hidden_brid_t + 1.0)
                brid_pen_t = torch.abs(torch.log(ratio))
            brid_pen_t = torch.clamp(brid_pen_t, 0.0, 5.0)
            brid_pen_t = torch.clamp(brid_pen_t, 0.0, 5.0)

            total_penalty = total_penalty + (bond_pen_t + brid_pen_t)

            # logs
            tot_exp_hidden_bond += float(exp_hidden_bond_t.item())
            tot_exp_hidden_brid += float(exp_hidden_brid_t.item())
            tot_pred_bond += float(pred_bond_t.item())
            tot_pred_brid += float(pred_brid_t.item())
            tot_cand_pairs += n_cand_t

        final_penalty = penalty_strength * (total_penalty / float(max_timestep + 1))

        if neighbor_index is not None:
            print(f"[SparsePenalty] avg candidate pairs per t ≈ {tot_cand_pairs / float(max_timestep + 1):.1f}")
        print(f"  Expected hidden (candidates): bonding={tot_exp_hidden_bond:.1f}, bridging={tot_exp_hidden_brid:.1f}")
        print(f"  Predicted (candidates):       bonding={tot_pred_bond:.1f}, bridging={tot_pred_brid:.1f}")
        print(f"  Final penalty: {float(final_penalty.item()):.3f}")

        return final_penalty


    
    # def compute_type_specific_density_penalty(self, marginal_probs, network_data, max_timestep,
    #                                           temperature=0.01, balance_factor=1.0, penalty_strength=1.0):
    #     """
    #     RECOMMENDED: Normalized log penalty with per-type weighting
        
    #     Key advantages for your case:
    #     1. Handles vastly different scales (12 vs 3800)
    #     2. Provides meaningful gradients regardless of magnitude
    #     3. Weights types by their relative importance
    #     4. Symmetric treatment of over/under-estimation
    #     """
    #     # Count observed edges
    #     observed_bonding = 0
    #     observed_bridging = 0
        
    #     for t in range(max_timestep + 1):
    #         for i, j, link_type in network_data.get_observed_edges_at_time(t):
    #             if link_type == 1:
    #                 observed_bonding += 1
    #             elif link_type == 2:
    #                 observed_bridging += 1
        
    #     # Estimate expected counts
    #     expected_hidden_bonding = max((observed_bonding / (1 - self.rho_1)) - observed_bonding, 1.0)
    #     expected_hidden_bridging = max((observed_bridging / (1 - self.rho_2)) - observed_bridging, 1.0)
    #     print(f"expected_hidden_bonding: {expected_hidden_bonding}, expected_hidden_bridging: {expected_hidden_bridging}")
        
    #     # Gumbel-Softmax discrete counting
    #     discrete_bonding = 0.0
    #     discrete_bridging = 0.0
        
    #     for pair_key, π_ij in marginal_probs.items():
    #         logits = torch.log(π_ij + 1e-8)
    #         sharp_probs = F.softmax(logits / temperature, dim=0)
            
    #         discrete_bonding += sharp_probs[1]
    #         discrete_bridging += sharp_probs[2]
        
    #     # Scale-aware penalty computation
    #     # For bonding (rare): use relative error with high sensitivity
    #     bonding_relative_error = torch.abs(discrete_bonding - expected_hidden_bonding) / (expected_hidden_bonding if expected_hidden_bonding > 0 else 1.0)
    #     bonding_penalty = balance_factor * bonding_relative_error  # Higher weight for rare type
        
    #     # For bridging (common): use log ratio to handle large numbers
    #     bridging_ratio = (discrete_bridging + 1) / (expected_hidden_bridging + 1)
    #     bridging_penalty = torch.abs(torch.log(bridging_ratio))
        
    #     # Combine with controlled weighting
    #     max_bonding_penalty = 20.0
    #     max_bridging_penalty = 3.0
    #     bonding_penalty = 3*torch.clamp(bonding_penalty, 0.0, max_bonding_penalty)/20
    #     bridging_penalty = torch.clamp(bridging_penalty, 0.0, max_bridging_penalty)
        
    #     # Equal weighting (since we already balanced via different penalty types)
    #     combined_penalty = 0.5 * bonding_penalty + 0.5 * bridging_penalty
        
    #     # Scale by strength parameter
    #     final_penalty = penalty_strength * combined_penalty
        
    #     print(f"Gumbel Density Penalty (T={temperature}, balance={balance_factor}):")
    #     print(f"  Expected: bonding={expected_hidden_bonding:.1f}, bridging={expected_hidden_bridging:.1f}")
    #     print(f"  Discrete: bonding={discrete_bonding:.1f}, bridging={discrete_bridging:.1f}")
    #     print(f"  Bonding penalty (relative): {bonding_penalty:.3f}")
    #     print(f"  Bridging penalty (log-ratio): {bridging_penalty:.3f}")
    #     print(f"  Final penalty: {final_penalty:.3f}")
        
    #     return final_penalty

    
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

        def get_dynamic_weights(epoch):
            if epoch < 150:  
                return {
                    'state': 0.0, 
                    'rollout': 0.0,
                    'observation': 0.1, 
                    # 'prior': 0.5, 
                    'prior': 1.0,       # syn_ruxiao_v3/syn_data1_200node
                    # 'prior': 1.0,     # syn_ruxiao_v2_data7_200node
                    # 'prior': 1.5,     # syn_ruxiao_v2_data7_200node test 2                    
                    'entropy': 1.0, 
                    'confidence': 0.5, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 2.0,       
                    'density_penalty': 1.5,   # syn_ruxiao_v3/syn_data1_200node             
                    # 'density_penalty': 1.0,   # syn_ruxiao_v2_200node
                    # 'density_penalty': 0.2,   # syn_abm_200node
                    # 'density_penalty': 1.5,
                    'info_propagation': 1e-5  # syn_ruxiao_v2_200node
                    # 'info_propagation': 2e-5   # syn_ruxiao_v2_data7_200node
                    # 'info_propagation': 1e-4  # syn_abm_200node
                    # 'info_propagation': 5e-5 
                }
            elif epoch < 350:
                progress = (epoch - 150) / 200
                state_weight = 0.1    
                prior_weight = 1.0 + 0.2 * progress 
                # prior_weight = 1.0 + 0.1 * progress    
                # prior_weight = 1.5  
                # density_weight = 2.0 * (1 - 0.5 * progress) 
                density_weight = 2.0
                return {
                    'state': state_weight, 
                    'rollout': 0.0,
                    'observation': 0.1, 
                    'prior': prior_weight,
                    'entropy': 1.0 - 0.3 * progress, 
                    'confidence': 0.5, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': density_weight,
                    # 'density_penalty': 0.2,   # syn_abm_200node
                    'density_penalty': 1.5,   # syn_ruxiao_v3/syn_data1_200node
                    # 'density_penalty': 1.0,  # syn_ruxiao_v2_200node
                    'info_propagation': 1e-5  # syn_ruxiao_v2_200node
                    # 'info_propagation': 2e-5   # syn_ruxiao_v2_data7_200node
                    # 'info_propagation': 5e-5   
                }
            elif epoch < 450:
                return {
                    'state': 0.1, 
                    'rollout': 0.0,
                    'observation': 0.1, 
                    'prior': 1.2,    # syn_ruxiao_v3/syn_data1_200node
                    # 'prior': 1.1,    # syn_ruxiao_v2_data7_200node
                    # 'prior': 1.5,    # syn_ruxiao_v2_data7_200node test 2
                    'entropy': 0.7, 
                    'confidence': 0.5, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 2.0, 
                    # 'density_penalty': 0.2,   # syn_abm_200node  
                    'density_penalty': 1.5,   # syn_ruxiao_v3/syn_data1_200node       
                    # 'density_penalty': 1.0,   # syn_ruxiao_v2_200node
                    # 'density_penalty': 1.5,
                    'info_propagation': 1e-5    # syn_ruxiao_v2_200node
                    # 'info_propagation': 2e-5   # syn_ruxiao_v2_data7_200node
                    # 'info_propagation': 5e-5
                }
            elif epoch < 550:  # NEW: bridging phase
                progress = (epoch - 450) / 100
                return {
                    'state': 0.1 + 0.4 * progress,      # 0.1 → 0.5
                    'rollout': 0.3 * progress,          # 0 → 0.3
                    'observation': 0.0, # 1.0 → 0
                    'prior': 0.0,      # 0.2 → 0
                    'entropy': 0.0,    # 0.7 → 0
                    'confidence': 0.0, # 0.5 → 0
                    'constraint': -lambda_constraint,
                    'density_bonus': 0.0,
                    'density_penalty': 0.0,
                    'info_propagation': 0.0
                }
            else:  
                return {
                    'state': 0.5, 
                    'rollout': 0.3,
                    'observation': 0.0, 
                    'prior': 0.0,
                    'entropy': 0.0, 
                    'confidence': 0.0, 
                    'constraint': -lambda_constraint, 
                    'density_bonus': 0.0,          
                    'density_penalty': 0.0,
                    'info_propagation': 0.0
                }
            
        import time
        timing_results = {}
        # Get normalization factors
        total_households = features.shape[0]
        total_possible_pairs = total_households * (total_households - 1) // 2
        total_timesteps = max_timestep + 1
        
        # Compute raw likelihoods (without any normalization first)
        start_time = time.time()
        state_likelihood_raw = self.compute_state_likelihood_batch(
            features, states, distances, node_batch, network_data, gumbel_samples, max_timestep, current_epoch=current_epoch
        )
        timing_results['state_likelihood'] = time.time() - start_time

        start_time = time.time()
        rollout_likelihood_raw = self.compute_rollout_state_likelihood_batch(
            features, states, distances, node_batch, network_data, gumbel_samples, max_timestep, current_epoch=current_epoch,
            rollout_steps=3, use_pred_prob=0.3
        )
        timing_results['rollout_likelihood'] = time.time() - start_time

        start_time = time.time()
        observation_likelihood_raw = self.compute_network_observation_likelihood_batch(
            marginal_probs, node_batch, network_data, max_timestep
        )
        timing_results['observation_likelihood'] = time.time() - start_time

        start_time = time.time()
        prior_likelihood_raw = self.compute_prior_likelihood_batch(
            conditional_probs, marginal_probs, features, states, distances, 
            node_batch, network_data, max_timestep
        )
        timing_results['prior_likelihood'] = time.time() - start_time

        # Compute other components
        start_time = time.time()
        posterior_entropy_raw = self.compute_posterior_entropy_batch(
            conditional_probs, marginal_probs, network_data, node_batch, max_timestep
        )
        timing_results['posterior_entropy'] = time.time() - start_time

        start_time = time.time()
        confidence_reg_raw = self.compute_confidence_regularization(marginal_probs)
        timing_results['confidence_regularization'] = time.time() - start_time

        # start_time = time.time()
        # constraint_penalty_raw = self.compute_constraint_penalty(
        # features, states, distances, node_batch, network_data, 
        # gumbel_samples, max_timestep
        # )
        # timing_results['constraint_penalty'] = time.time() - start_time

        start_time = time.time()
        timestep_density_penalty = self.compute_type_specific_density_penalty(
            marginal_probs, network_data, max_timestep
        )
        timing_results['timestep_density_penalty'] = time.time() - start_time

        start_time = time.time()
        info_propagation_penalty = self.compute_information_propagation_penalty(
        marginal_probs, network_data, max_timestep
        )
        timing_results['info_propagation_penalty'] = time.time() - start_time

        # density_bonus_raw = self.compute_connection_density_bonus(marginal_probs)

        print(f"Raw values - State: {state_likelihood_raw.item():.2f}, "
            f"Obs: {observation_likelihood_raw.item():.2f}, "
            f"Prior: {prior_likelihood_raw.item():.2f}, "
            f"Entropy: {posterior_entropy_raw.item():.2f},"
            f" Sparsity: {confidence_reg_raw.item():.2f}, "
            # f"Constraint: {constraint_penalty_raw.item():.2f}, "
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
        weighted_state_likelihood = weight['state'] * state_likelihood_per_prediction
        total_rollout_predictions = len(node_batch) * 3 * 3  # Assume 3-step rollout
        rollout_likelihood_per_prediction = rollout_likelihood_raw / total_rollout_predictions if total_rollout_predictions > 0 else torch.tensor(0.0)
        weighted_rollout_likelihood = weight.get('rollout', 0.0) * rollout_likelihood_per_prediction 
        weighted_observation_likelihood = weight['observation'] * network_likelihood_per_pair
        weighted_prior_likelihood = weight['prior'] * prior_likelihood_per_pair
        weighted_posterior_entropy = weight['entropy'] * entropy_per_pair
        weighted_confidence_reg = weight['confidence'] * confidence_per_pair
        # weighted_constraint_penalty = weight['constraint'] * constraint_penalty_raw
        weighted_density_penalty = weight['density_penalty'] * timestep_density_penalty
        weighted_info_propagation = weight['info_propagation'] * info_propagation_penalty

        # weighted_density_bonus = weight['density_bonus'] * density_bonus_raw

        # Total weighted ELBO
        # total_elbo = (weighted_state_likelihood + weighted_observation_likelihood +
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg - weighted_constraint_penalty)
        # total_elbo = (weighted_state_likelihood + weighted_observation_likelihood +
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg -
        #             weighted_constraint_penalty - weighted_density_penalty - weighted_info_propagation)
        total_elbo = (weighted_state_likelihood + weighted_rollout_likelihood + weighted_observation_likelihood +
                    weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg 
                     - weighted_density_penalty - weighted_info_propagation)

        print(f"Weighted components - State: {weighted_state_likelihood.item():.4f}, "
            f"Rollout: {weighted_rollout_likelihood.item():.4f}, "
            f"Obs: {weighted_observation_likelihood.item():.4f}, "
            f"Prior: {weighted_prior_likelihood.item():.4f}, "
            f"Entropy: {weighted_posterior_entropy.item():.4f}")
        print(f"Sparsity: {weighted_confidence_reg.item():.4f}, "
            # f"Constraint: {weighted_constraint_penalty.item():.4f}, "
            f"Timestep Density Penalty: {weighted_density_penalty.item():.4f}, "
            f"Info Propagation: {weighted_info_propagation.item():.4f}")
            # f"Density Bonus: {weighted_density_bonus.item():.4f}")
        print(f"Total weighted ELBO: {total_elbo.item():.4f}")

        print("\n=== FUNCTION TIMING ANALYSIS ===")
        total_time = sum(timing_results.values())
        for func_name, exec_time in timing_results.items():
            percentage = (exec_time / total_time) * 100 if total_time > 0 else 0
            print(f"{func_name:25s}: {exec_time:.4f}s ({percentage:5.1f}%)")
        print(f"{'TOTAL TIME':25s}: {total_time:.4f}s (100.0%)")

        return {
            'state_likelihood': weighted_state_likelihood,
            'rollout_likelihood': weighted_rollout_likelihood,
            'observation_likelihood': weighted_observation_likelihood, 
            'prior_likelihood': weighted_prior_likelihood,
            'posterior_entropy': weighted_posterior_entropy,
            'confidence_regularization': weighted_confidence_reg,
            # 'constraint_penalty': weighted_constraint_penalty,
            'density_penalty': weighted_density_penalty,
            'info_propagation_penalty': weighted_info_propagation,
            # 'density_bonus': weighted_density_bonus,
            'total_elbo': total_elbo
        }