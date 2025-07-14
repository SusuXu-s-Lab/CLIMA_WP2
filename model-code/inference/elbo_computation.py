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
                 rho_1: float = 0.5, rho_2: float = 0.5, confidence_weight: float = 0.0):
        self.network_evolution = network_evolution
        self.state_transition = state_transition
        self.confidence_weight = confidence_weight
        
        # Observation model parameters
        self.rho_1 = torch.nn.Parameter(torch.tensor(rho_1))  # Probability of observing link type 1
        self.rho_2 = torch.nn.Parameter(torch.tensor(rho_2))  # Probability of observing link type 2

    
    # def compute_state_likelihood_batch(self, features: torch.Tensor,
    #                                     states: torch.Tensor, distances: torch.Tensor,
    #                                     node_batch: torch.Tensor, network_data,
    #                                     gumbel_samples: List[Dict[str, torch.Tensor]], max_timestep: int,
    #                                 ) -> torch.Tensor:
    #     """
    #     Computes the log-likelihood of household state transitions for a batch of nodes,
    #     averaged over multiple Gumbel-Softmax samples.

    #     Key optimizations
    #     -----------------
    #     1.  All tensors that do NOT depend on the current Gumbel sample are pre-computed
    #         once outside the sample loop (undecided masks, labels, class weights, etc.).
    #     2.  Inside the sample loop, the only heavy call is
    #         `compute_activation_probability`, which IS sample-dependent.
    #     3.  Log-probabilities are computed in a fully vectorized form.
    #     """
    #     if not gumbel_samples:                             # no samples → zero likelihood
    #         return torch.tensor(0.0, device=features.device)

    #     device     = features.device
    #     batch_idx  = node_batch                           # alias
    #     B          = batch_idx.numel()
    #     S          = len(gumbel_samples)

    #     # ------------------------------------------------------------
    #     # 1. Pre-processing (sample-independent)
    #     # ------------------------------------------------------------
    #     # states shape: [N, T_total + 1, 3]
    #     batch_states   = states[batch_idx]                # [B, T_total + 1, 3]

    #     undecided_mask = batch_states[:, :max_timestep] == 0        # [B, T, 3]
    #     labels         = batch_states[:, 1 : max_timestep + 2]      # [B, T, 3]
    #     print(f"undecided_mask shape: {undecided_mask.shape}, labels shape: {labels.shape}")

    #     static_entries = []   # each entry: (t, k, undecided_ids, label_tensor, pos_w, neg_w)

    #     for t in range(max_timestep):
    #         for k in range(3):
    #             mask = undecided_mask[:, t, k]            # [B]
    #             if not mask.any():
    #                 continue

    #             undecided_ids = batch_idx[mask]           # global household ids, 1-D
    #             y             = labels[mask, t+1, k].float()  # 0/1 labels, 1-D

    #             n_pos   = y.sum()
    #             n_total = y.numel()
    #             if n_pos > 0:
    #                 pos_w = min(5.0 * (n_total - n_pos) / n_pos, 100.0)
    #             else:
    #                 pos_w = 1.0
    #             neg_w = 1.0

    #             static_entries.append(
    #                 (t, k, undecided_ids, y,
    #                 pos_w if isinstance(pos_w, torch.Tensor) else torch.tensor(pos_w, device=device),
    #                 neg_w if isinstance(neg_w, torch.Tensor) else torch.tensor(neg_w, device=device))
    #             )

    #     if not static_entries:                           # nothing to score
    #         return torch.tensor(0.0, device=device)

    #     # ------------------------------------------------------------
    #     # 2. Loop over Gumbel samples
    #     # ------------------------------------------------------------
    #     total_ll = 0.0
    #     log = torch.log
    #     log1p = torch.log1p

    #     for sample in gumbel_samples:
    #         sample_ll = 0.0

    #         for t, k, undec_ids, y, pos_w, neg_w in static_entries:
    #             # activation probabilities, shape [n_u]
    #             p = self.state_transition.compute_activation_probability(
    #                 household_idx = undec_ids,
    #                 decision_type = k,
    #                 features      = features,
    #                 states        = states,
    #                 distances     = distances,
    #                 network_data  = network_data,
    #                 gumbel_samples= sample,
    #                 time          = t,
    #             )

    #             print(f"p(t={t}, k={k}, undec_ids={undec_ids.tolist()}): {p.tolist()}")

    #             # safe logs
    #             log_p     = log(p.clamp_min(1e-8))
    #             log_not_p = log1p(-p.clamp_max(1.0 - 1e-8))

    #             sample_ll += (
    #                 pos_w * (y * log_p).sum() +
    #                 neg_w * ((1.0 - y) * log_not_p).sum()
    #             )

    #         total_ll += sample_ll

    #     return total_ll / S

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
            
            pos_weight = min(5*(n_total - n_positive) / n_positive, 100.0)
            return pos_weight, 1.0

        total_likelihood = 0.0
        num_samples = len(gumbel_samples)
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
        
        return total_likelihood/num_samples
      

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

    def compute_prior_likelihood_batch_optimized(self,
                                    conditional_probs: Dict[str, torch.Tensor],
                                    marginal_probs: Dict[str, torch.Tensor],
                                    features: torch.Tensor,
                                    states: torch.Tensor,
                                    distances: torch.Tensor,
                                    node_batch: torch.Tensor,
                                    network_data,
                                    max_timestep: int) -> torch.Tensor:
            """
            OPTIMIZED: Prior likelihood for COMPLETE graph with batch processing.
            
            Key optimizations:
            1. Pre-compute all batch pairs once
            2. Vectorized neural network calls
            3. Batch feature extraction
            4. Reduced repeated computations
            """
            total_likelihood = 0.0
            batch_nodes_set = set(node_batch.tolist())
            
            # PRE-COMPUTE: Get ALL batch pairs for all timesteps at once
            def get_all_batch_pairs_optimized():
                """Pre-compute all pairs involving batch nodes for efficiency"""
                all_pairs = []
                n_nodes = features.shape[0]
                
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        if i in batch_nodes_set or j in batch_nodes_set:
                            all_pairs.append((i, j))
                
                return all_pairs
            
            all_batch_pairs = get_all_batch_pairs_optimized()
            
            if len(all_batch_pairs) == 0:
                return torch.tensor(0.0)
            
            # OPTIMIZATION 1: Batch process initial probabilities (t=0)
            initial_pairs = []
            initial_marginals = []
            initial_observed_types = []
            
            for i, j in all_batch_pairs:
                if network_data.is_observed(i, j, 0):
                    observed_type = network_data.get_link_type(i, j, 0)
                    initial_observed_types.append((i, j, observed_type))
                else:
                    pair_key = f"{i}_{j}_0"
                    if pair_key in marginal_probs:
                        marginal_tensor = marginal_probs[pair_key]
                        initial_pairs.append((i, j))
                        initial_marginals.append(marginal_tensor)
            
            # Batch compute initial probabilities for hidden pairs
            if initial_pairs:
                batch_size = len(initial_pairs)
                i_indices = [pair[0] for pair in initial_pairs]
                j_indices = [pair[1] for pair in initial_pairs]
                
                # Ensure all marginals are 3-dimensional
                validated_marginals = []
                for marginal in initial_marginals:
                    if marginal.dim() == 1 and marginal.shape[0] == 3:
                        validated_marginals.append(marginal)
                    else:
                        continue  # Skip invalid marginals
                
                if len(validated_marginals) != len(initial_pairs):
                    # Update indices to match validated marginals
                    initial_pairs = initial_pairs[:len(validated_marginals)]
                    i_indices = i_indices[:len(validated_marginals)]
                    j_indices = j_indices[:len(validated_marginals)]
                
                if len(validated_marginals) > 0:
                    # Stack marginals
                    marginals_batch = torch.stack(validated_marginals)  # [batch_size, 3]
                
                    # Vectorized feature extraction
                    feat_i_batch = features[i_indices]  # [batch_size, feature_dim]
                    feat_j_batch = features[j_indices]  # [batch_size, feature_dim]
                    dist_batch = distances[i_indices, j_indices].unsqueeze(1)  # [batch_size, 1]
                    
                    # Single batch call to network evolution model
                    init_probs_batch = self.network_evolution.initial_probabilities(
                        feat_i_batch, feat_j_batch, dist_batch
                    )  # [batch_size, 3] or [batch_size, 3, 1]
                    
                    # Fix shape mismatch: squeeze any trailing dimensions
                    init_probs_batch = init_probs_batch.squeeze(-1)  # Ensure [batch_size, 3]
                    
                    # Vectorized log probability computation
                    log_probs_batch = torch.sum(marginals_batch * torch.log(init_probs_batch + 1e-8), dim=1)
                    total_likelihood += log_probs_batch.sum()
            
            # Handle observed pairs at t=0
            for i, j, observed_type in initial_observed_types:
                π_ij = F.one_hot(torch.tensor(observed_type), num_classes=3).float()
                feat_i = features[i].unsqueeze(0)
                feat_j = features[j].unsqueeze(0)
                dist = distances[i, j].unsqueeze(0).unsqueeze(1)
                
                init_probs = self.network_evolution.initial_probabilities(feat_i, feat_j, dist)
                log_prob = torch.sum(π_ij * torch.log(init_probs.squeeze(0) + 1e-8))
                total_likelihood += log_prob
            
            # OPTIMIZATION 2: Batch process temporal transitions
            for t in range(1, max_timestep + 1):
                # Group pairs by case type for batch processing
                hidden_to_hidden_pairs = []
                hidden_to_obs_pairs = []
                obs_to_hidden_pairs = []
                obs_to_obs_pairs = []
                
                for i, j in all_batch_pairs:
                    current_observed = network_data.is_observed(i, j, t)
                    prev_observed = network_data.is_observed(i, j, t-1)
                    
                    if not current_observed and not prev_observed:
                        hidden_to_hidden_pairs.append((i, j))
                    elif current_observed and not prev_observed:
                        hidden_to_obs_pairs.append((i, j))
                    elif not current_observed and prev_observed:
                        obs_to_hidden_pairs.append((i, j))
                    else:
                        obs_to_obs_pairs.append((i, j))
                
                # Process each case type in batches
                total_likelihood += self._process_hidden_to_hidden_batch(
                    hidden_to_hidden_pairs, conditional_probs, marginal_probs, 
                    features, states, distances, t
                )
                
                total_likelihood += self._process_hidden_to_obs_batch(
                    hidden_to_obs_pairs, marginal_probs, features, states, distances, 
                    network_data, t
                )
                
                total_likelihood += self._process_obs_to_hidden_batch(
                    obs_to_hidden_pairs, marginal_probs, features, states, distances, 
                    network_data, t
                )
                
                total_likelihood += self._process_obs_to_obs_batch(
                    obs_to_obs_pairs, features, states, distances, network_data, t
                )
            
            return total_likelihood
    
    def _process_hidden_to_hidden_batch(self, pairs, conditional_probs, marginal_probs, 
                                       features, states, distances, t):
        """Batch process hidden→hidden transitions"""
        if not pairs:
            return torch.tensor(0.0)
        
        total_likelihood = 0.0
        valid_pairs = []
        valid_conditionals = []
        valid_prev_marginals = []
        
        # Filter valid pairs
        for i, j in pairs:
            pair_key_current = f"{i}_{j}_{t}"
            pair_key_prev = f"{i}_{j}_{t-1}"
            
            if (pair_key_current in conditional_probs and 
                pair_key_current in marginal_probs and 
                pair_key_prev in marginal_probs):
                
                valid_pairs.append((i, j))
                valid_conditionals.append(conditional_probs[pair_key_current])
                valid_prev_marginals.append(marginal_probs[pair_key_prev])
        
        if not valid_pairs:
            return torch.tensor(0.0)
        
        # Batch compute transition probabilities
        batch_size = len(valid_pairs)
        i_indices = [pair[0] for pair in valid_pairs]
        j_indices = [pair[1] for pair in valid_pairs]
        
        feat_i_batch = features[i_indices]
        feat_j_batch = features[j_indices]
        state_i_batch = states[i_indices, t]
        state_j_batch = states[j_indices, t]
        dist_batch = distances[i_indices, j_indices].unsqueeze(1)
        
        # Process in chunks to manage memory
        chunk_size = 32  # Adjust based on available memory
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            
            chunk_prev_marginals = valid_prev_marginals[chunk_start:chunk_end]
            chunk_conditionals = valid_conditionals[chunk_start:chunk_end]
            
            # Validate marginal shapes
            validated_chunk_marginals = []
            for marginal in chunk_prev_marginals:
                if marginal.dim() == 1 and marginal.shape[0] == 3:
                    validated_chunk_marginals.append(marginal)
                else:
                    continue
            
            if len(validated_chunk_marginals) == 0:
                continue
                
            # Expand previous marginals for transition computation
            π_prev_batch = torch.stack(validated_chunk_marginals).unsqueeze(1)  # [chunk_size, 1, 3]
            
            # Adjust batch indices to match validated marginals
            valid_chunk_size = len(validated_chunk_marginals)
            
            trans_probs_batch = self.network_evolution.transition_probabilities(
                π_prev_batch, 
                feat_i_batch[chunk_start:chunk_start + valid_chunk_size], 
                feat_j_batch[chunk_start:chunk_start + valid_chunk_size],
                state_i_batch[chunk_start:chunk_start + valid_chunk_size], 
                state_j_batch[chunk_start:chunk_start + valid_chunk_size], 
                dist_batch[chunk_start:chunk_start + valid_chunk_size]
            )  # [valid_chunk_size, 3, 3]
            
            # Vectorized complex formula computation
            for idx in range(valid_chunk_size):
                π_prev = validated_chunk_marginals[idx]  # [3]
                π_conditional = chunk_conditionals[idx]  # [3, 3]
                trans_probs = trans_probs_batch[idx]  # [3, 3]
                
                # Vectorized computation of double sum
                prob_contrib = torch.sum(
                    π_prev.unsqueeze(1) * π_conditional * torch.log(trans_probs + 1e-8)
                )
                total_likelihood += prob_contrib
        
        return total_likelihood
    
    def _process_hidden_to_obs_batch(self, pairs, marginal_probs, features, states, 
                                    distances, network_data, t):
        """Batch process hidden→observed transitions"""
        if not pairs:
            return torch.tensor(0.0)
        
        # Group by observed type for more efficient batch processing
        type_groups = {0: [], 1: [], 2: []}
        
        for i, j in pairs:
            pair_key_prev = f"{i}_{j}_{t-1}"
            if pair_key_prev in marginal_probs:
                k_obs = network_data.get_link_type(i, j, t)
                # Ensure k_obs is within valid range
                if k_obs in {0, 1, 2}:
                    type_groups[k_obs].append((i, j, marginal_probs[pair_key_prev]))
                else:
                    continue  # Skip invalid link types
        
        total_likelihood = 0.0
        
        for k_obs, group_pairs in type_groups.items():
            if not group_pairs:
                continue
            
            batch_size = len(group_pairs)
            i_indices = [i for i, j, _ in group_pairs]
            j_indices = [j for i, j, _ in group_pairs] 
            prev_marginals = [marginal for i, j, marginal in group_pairs]
            
            # Batch feature extraction
            feat_i_batch = features[i_indices]
            feat_j_batch = features[j_indices]
            state_i_batch = states[i_indices, t]
            state_j_batch = states[j_indices, t]
            dist_batch = distances[i_indices, j_indices].unsqueeze(1)
            
            # Compute transition probabilities for all 3 previous types
            prev_types_all = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 3, 3]
            
            trans_probs = self.network_evolution.transition_probabilities(
                prev_types_all.reshape(-1, 3),  # [batch_size*3, 3]
                feat_i_batch.repeat(3, 1),   # [batch_size*3, feature_dim]
                feat_j_batch.repeat(3, 1),   # [batch_size*3, feature_dim]
                state_i_batch.repeat(3, 1),  # [batch_size*3, 3]
                state_j_batch.repeat(3, 1),  # [batch_size*3, 3]
                dist_batch.repeat(3, 1)      # [batch_size*3, 1]
            ).reshape(batch_size, 3, 3, 3)      # [batch_size, 3, 3, 3]
            
            # Extract relevant transitions and compute likelihood
            for idx, (π_prev) in enumerate(prev_marginals):
                # Validate π_prev shape
                if π_prev.dim() != 1 or π_prev.shape[0] != 3:
                    continue
                    
                transition_to_obs = trans_probs[idx, torch.arange(3), torch.arange(3), k_obs]
                log_probs = torch.log(transition_to_obs + 1e-8)
                prob_contrib = torch.sum(π_prev * log_probs)
                total_likelihood += prob_contrib
        
        return total_likelihood
    
    def _process_obs_to_hidden_batch(self, pairs, marginal_probs, features, states, 
                                    distances, network_data, t):
        """Batch process observed→hidden transitions"""
        if not pairs:
            return torch.tensor(0.0)
        
        # Group by previous observed type
        type_groups = {0: [], 1: [], 2: []}
        
        for i, j in pairs:
            pair_key_current = f"{i}_{j}_{t}"
            if pair_key_current in marginal_probs:
                k_prev_obs = network_data.get_link_type(i, j, t-1)
                # Ensure k_prev_obs is within valid range
                if k_prev_obs in {0, 1, 2}:
                    type_groups[k_prev_obs].append((i, j, marginal_probs[pair_key_current]))
                else:
                    continue  # Skip invalid link types
        
        total_likelihood = 0.0
        
        for k_prev_obs, group_pairs in type_groups.items():
            if not group_pairs:
                continue
            
            batch_size = len(group_pairs)
            i_indices = [i for i, j, _ in group_pairs]
            j_indices = [j for i, j, _ in group_pairs]
            current_marginals = [marginal for i, j, marginal in group_pairs]
            
            # Batch feature extraction
            feat_i_batch = features[i_indices]
            feat_j_batch = features[j_indices]
            state_i_batch = states[i_indices, t]
            state_j_batch = states[j_indices, t]
            dist_batch = distances[i_indices, j_indices].unsqueeze(1)
            
            # Create previous type tensor for batch
            prev_onehot = F.one_hot(torch.tensor(k_prev_obs), num_classes=3).float()
            prev_onehot_batch = prev_onehot.unsqueeze(0).expand(batch_size, -1)
            
            # Batch compute transition probabilities
            trans_probs = self.network_evolution.transition_probabilities(
                prev_onehot_batch, feat_i_batch, feat_j_batch, 
                state_i_batch, state_j_batch, dist_batch
            )  # [batch_size, 3, 3]
            
            # Compute likelihood for each pair
            for idx, π_current in enumerate(current_marginals):
                # Validate π_current shape
                if π_current.dim() != 1 or π_current.shape[0] != 3:
                    continue
                    
                for k_curr in range(3):
                    trans_prob = trans_probs[idx, k_prev_obs, k_curr]
                    prob_contrib = π_current[k_curr] * torch.log(trans_prob + 1e-8)
                    total_likelihood += prob_contrib
        
        return total_likelihood
    
    def _process_obs_to_obs_batch(self, pairs, features, states, distances, network_data, t):
        """Batch process observed→observed transitions"""
        if not pairs:
            return torch.tensor(0.0)
        
        # Group by transition type for batch processing
        transition_groups = {}
        
        for i, j in pairs:
            k_prev_obs = network_data.get_link_type(i, j, t-1)
            k_curr_obs = network_data.get_link_type(i, j, t)
            
            # Ensure both link types are within valid range
            if k_prev_obs not in {0, 1, 2} or k_curr_obs not in {0, 1, 2}:
                continue  # Skip invalid link types
                
            transition_key = (k_prev_obs, k_curr_obs)
            
            if transition_key not in transition_groups:
                transition_groups[transition_key] = []
            transition_groups[transition_key].append((i, j))
        
        total_likelihood = 0.0
        
        for (k_prev_obs, k_curr_obs), group_pairs in transition_groups.items():
            batch_size = len(group_pairs)
            i_indices = [pair[0] for pair in group_pairs]
            j_indices = [pair[1] for pair in group_pairs]
            
            # Batch feature extraction
            feat_i_batch = features[i_indices]
            feat_j_batch = features[j_indices]
            state_i_batch = states[i_indices, t]
            state_j_batch = states[j_indices, t]
            dist_batch = distances[i_indices, j_indices].unsqueeze(1)
            
            # Create previous type tensor for batch
            prev_onehot = F.one_hot(torch.tensor(k_prev_obs), num_classes=3).float()
            prev_onehot_batch = prev_onehot.unsqueeze(0).expand(batch_size, -1)
            
            # Batch compute transition probabilities
            trans_probs = self.network_evolution.transition_probabilities(
                prev_onehot_batch, feat_i_batch, feat_j_batch, 
                state_i_batch, state_j_batch, dist_batch
            )  # [batch_size, 3, 3]
            
            # Extract specific transition probabilities and sum log probabilities
            specific_trans_probs = trans_probs[:, k_prev_obs, k_curr_obs]  # [batch_size]
            log_probs = torch.log(specific_trans_probs + 1e-8)
            total_likelihood += log_probs.sum()
        
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

    def compute_posterior_entropy_batch_optimized(self,
                                                conditional_probs: Dict[str, torch.Tensor],
                                                marginal_probs: Dict[str, torch.Tensor],
                                                network_data,
                                                node_batch: torch.Tensor,
                                                max_timestep: int) -> torch.Tensor:
        """
        OPTIMIZED: Posterior entropy computation with batched processing and reduced string operations.
        
        Optimization strategies:
        1. Pre-index all relevant probability tensors by timestep
        2. Batch vectorized computations where possible
        3. Reduce string concatenation and dictionary lookups
        4. Memory-efficient chunked processing
        """
        total_entropy = 0.0
        batch_nodes_set = set(node_batch.tolist())
        
        # Pre-processing: Organize data by timestep to reduce repeated lookups
        timestep_data = {}
        for t in range(max_timestep + 1):
            timestep_data[t] = {
                'hidden_pairs': [],
                'marginal_tensors': [],
                'conditional_tensors': [],
                'prev_observed_types': []
            }
        
        # Batch collect data - single pass through marginal_probs
        for pair_key, marginal_prob in marginal_probs.items():
            parts = pair_key.split('_')
            if len(parts) == 3:
                i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Only process pairs involving batch nodes
                if i in batch_nodes_set or j in batch_nodes_set:
                    if t < len(timestep_data):
                        timestep_data[t]['hidden_pairs'].append((i, j))
                        timestep_data[t]['marginal_tensors'].append(marginal_prob)
        
        # Batch compute initial entropy (t=0)
        if timestep_data[0]['marginal_tensors']:
            marginal_batch = torch.stack(timestep_data[0]['marginal_tensors'])  # [N, 3]
            
            # Vectorized entropy computation: H = -Σ p * log(p)
            log_marginals = torch.log(marginal_batch + 1e-8)
            entropy_batch = -torch.sum(marginal_batch * log_marginals, dim=1)
            total_entropy += entropy_batch.sum()
        
        # Batch compute temporal entropy (t>=1)
        for t in range(1, max_timestep + 1):
            # Collect conditional probabilities for current timestep
            current_conditionals = []
            prev_marginals = []
            obs_conditionals = []
            
            for i, j in timestep_data[t]['hidden_pairs']:
                pair_key_current = f"{i}_{j}_{t}"
                pair_key_prev = f"{i}_{j}_{t-1}"
                
                if pair_key_current in conditional_probs:
                    π_conditional = conditional_probs[pair_key_current]
                    
                    if network_data.is_observed(i, j, t-1):
                        # Previous state observed case
                        prev_type = network_data.get_link_type(i, j, t-1)
                        obs_conditionals.append(π_conditional[prev_type, :])
                    elif pair_key_prev in marginal_probs:
                        # Previous state hidden case
                        current_conditionals.append(π_conditional)
                        prev_marginals.append(marginal_probs[pair_key_prev])
            
            # Batch process observed conditional entropy
            if obs_conditionals:
                obs_batch = torch.stack(obs_conditionals)  # [N, 3]
                log_obs = torch.log(obs_batch + 1e-8)
                entropy_obs = -torch.sum(obs_batch * log_obs, dim=1)
                total_entropy += entropy_obs.sum()
            
            # Batch process hidden conditional entropy (still requires loop but vectorized inner computation)
            if current_conditionals and prev_marginals:
                for π_conditional, π_prev in zip(current_conditionals, prev_marginals):
                    # Vectorized double summation: H = -Σ Σ [π̄(t-1)[k'] × π(t|k')[k]] × log π(t|k')[k]
                    π_prev_expanded = π_prev.unsqueeze(1)  # [3, 1]
                    joint_probs = π_prev_expanded * π_conditional  # [3, 3]
                    
                    # Safe logarithm computation
                    log_conditional = torch.log(π_conditional + 1e-8)
                    valid_mask = joint_probs > 1e-8
                    
                    entropy_contribution = -torch.sum(
                        joint_probs * log_conditional * valid_mask.float()
                    )
                    total_entropy += entropy_contribution
        
        return total_entropy
    
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
            
            expected_hidden_bonding_t = max(expected_total_bonding_t - observed_bonding_t, 0.0)
            expected_hidden_bridging_t = max(expected_total_bridging_t - observed_bridging_t, 0.0)

            # Convert to tensor only when needed
            expected_hidden_bonding_t = torch.tensor(expected_hidden_bonding_t, dtype=torch.float32)
            expected_hidden_bridging_t = torch.tensor(expected_hidden_bridging_t, dtype=torch.float32)
            
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
            
            ε = 1e-6
            if expected_hidden_bonding_t > 0.1:
                bonding_penalty_t = torch.abs(
                    discrete_bonding_t - expected_hidden_bonding_t) / (expected_hidden_bonding_t + ε)
            else:      
                bonding_penalty_t = torch.tensor(0.0, device=discrete_bonding_t.device)

            # --- bridging penalty ------------------------------------------------

            if expected_hidden_bridging_t > 0.1:
                bridging_penalty_t = torch.abs(
                    discrete_bridging_t - expected_hidden_bridging_t) / (expected_hidden_bridging_t + ε)
            else:
                bridging_penalty_t = torch.tensor(0.0, device=discrete_bridging_t.device)
            
            # Combine penalties for current timestep
            # timestep_penalty = 0.5 * bonding_penalty_t + 0.5 * bridging_penalty_t
            timestep_penalty = 0.7 * bonding_penalty_t + 0.3 * bridging_penalty_t

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
        import time

        # def get_dynamic_weights(epoch):
        #     if epoch < 30:  
        #         return {
        #             'state': 1.0, 'observation': 1.0, 'prior': 0.001,          
        #             'entropy': 1.0, 'confidence': 0, 'constraint': -lambda_constraint, 'density_bonus': 1.0
        #         }
        #     elif epoch < 100:  # 50-100 epoch: introduce prior gradually
        #         progress = (epoch - 30) / 70  # 0 to 1
        #         prior_weight = 0.001 + 0.099 * progress
        #         return {
        #             'state': 1.0, 'observation': 1.0, 'prior': prior_weight ,
        #             'entropy': 1.0 - 0.3 * progress, 'confidence': 0, 'constraint': -lambda_constraint, 'density_bonus': 1.0
        #         }
        #     else:  # 100 epoch: full weight
        #         return {
        #             'state': 1.0, 'observation': 1.0, 'prior': 0.1,
        #             'entropy': 0.7, 'confidence': 0, 'constraint': -lambda_constraint, 'density_bonus': 1.0
        #         }

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
                    'info_propagation': 1e-4   
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
            
        import time
        timing_results = {}
        # Get normalization factors
        total_households = features.shape[0]
        total_possible_pairs = total_households * (total_households - 1) // 2
        total_timesteps = max_timestep + 1
        
        # Compute raw likelihoods (without any normalization first)
        start_time = time.time()
        state_likelihood_raw = self.compute_state_likelihood_batch(
            features, states, distances, node_batch, network_data, gumbel_samples, max_timestep
        )
        timing_results['state_likelihood'] = time.time() - start_time

        start_time = time.time()
        observation_likelihood_raw = self.compute_network_observation_likelihood_batch(
            marginal_probs, node_batch, network_data, max_timestep
        )
        timing_results['observation_likelihood'] = time.time() - start_time

        start_time = time.time()
        prior_likelihood_raw = self.compute_prior_likelihood_batch_optimized(
            conditional_probs, marginal_probs, features, states, distances, 
            node_batch, network_data, max_timestep
        )
        timing_results['prior_likelihood'] = time.time() - start_time

        # Compute other components
        start_time = time.time()
        posterior_entropy_raw = self.compute_posterior_entropy_batch_optimized(
            conditional_probs, marginal_probs, network_data, node_batch, max_timestep
        )
        timing_results['posterior_entropy'] = time.time() - start_time

        start_time = time.time()
        confidence_reg_raw = self.compute_confidence_regularization(marginal_probs)
        timing_results['confidence_regularization'] = time.time() - start_time

        start_time = time.time()
        constraint_penalty_raw = self.compute_constraint_penalty(
        features, states, distances, node_batch, network_data, 
        gumbel_samples, max_timestep
        )
        timing_results['constraint_penalty'] = time.time() - start_time

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

        # Helper function to safely extract numeric values
        def safe_item(x):
            return x.item() if hasattr(x, 'item') else x
        
        print(f"Raw values - State: {safe_item(state_likelihood_raw):.2f}, "
            f"Obs: {safe_item(observation_likelihood_raw):.2f}, "
            f"Prior: {safe_item(prior_likelihood_raw):.2f}, "
            f"Entropy: {safe_item(posterior_entropy_raw):.2f},"
            f" Sparsity: {safe_item(confidence_reg_raw):.2f}, "
            f"Constraint: {safe_item(constraint_penalty_raw):.2f}, "
            f"Timestep Density Penalty: {safe_item(timestep_density_penalty):.2f}, "
            f"Info Propagation Penalty: {safe_item(info_propagation_penalty):.2f},")
            #f"Density Bonus: {safe_item(density_bonus_raw):.2f}")

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

        print(f"Weighted components - State: {safe_item(weighted_state_likelihood):.4f}, "
            f"Obs: {safe_item(weighted_observation_likelihood):.4f}, "
            f"Prior: {safe_item(weighted_prior_likelihood):.4f}, "
            f"Entropy: {safe_item(weighted_posterior_entropy):.4f}")
        print(f"Sparsity: {safe_item(weighted_confidence_reg):.4f}, "
            f"Constraint: {safe_item(weighted_constraint_penalty):.4f}, "
            f"Timestep Density Penalty: {safe_item(weighted_density_penalty):.4f}, "
            f"Info Propagation: {safe_item(weighted_info_propagation):.4f}")
            # f"Density Bonus: {safe_item(weighted_density_bonus):.4f}")
        print(f"Total weighted ELBO: {safe_item(total_elbo):.4f}")

        print("\n=== FUNCTION TIMING ANALYSIS ===")
        total_time = sum(timing_results.values())
        for func_name, exec_time in timing_results.items():
            percentage = (exec_time / total_time) * 100 if total_time > 0 else 0
            print(f"{func_name:25s}: {exec_time:.4f}s ({percentage:5.1f}%)")
        print(f"{'TOTAL TIME':25s}: {total_time:.4f}s (100.0%)")

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