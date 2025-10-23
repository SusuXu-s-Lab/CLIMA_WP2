import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple
from models import NetworkEvolution, StateTransition
from collections import defaultdict

def adaptive_focal_loss(p, y, timestep, total_timesteps, base_alpha=0.85, base_gamma=2.5):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    
    if n_pos == 0:
        return -torch.log(1 - p.clamp(1e-6, 1-1e-6)).mean()
    
    ratio = n_neg / n_pos
    alpha = 1 - 1 / (1 + 0.1 * ratio)
    alpha = torch.clamp(alpha, base_alpha, 0.99)
    
    time_boost = 1 + 0.5 * (timestep / total_timesteps)
    alpha = torch.clamp(alpha * time_boost, base_alpha, 0.99)
    
    gamma = base_gamma + 0.5 * torch.log10(torch.clamp(ratio, 1, 1000))
    gamma = torch.clamp(gamma, 2.0, 3.0)

    eps = 1e-6
    p = torch.clamp(p, eps, 1-eps)
    loss_pos = -alpha * (1 - p)**gamma * y * torch.log(p)
    loss_neg = -(1 - alpha) * p**gamma * (1 - y) * torch.log(1 - p)

    loss = (loss_pos + loss_neg).mean()

    assert not torch.isnan(loss) and not torch.isinf(loss), \
        f"Focal loss output NaN/Inf at timestep={timestep}, " \
        f"alpha={alpha:.3f}, gamma={gamma:.3f}, " \
        f"p range=[{p.min():.6f}, {p.max():.6f}]"
    
    return loss



class ELBOComputation(nn.Module):
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
                 rho_1: float = 0.5, rho_2: float = 0.5, confidence_weight: float = 0.0, variational_posterior=None):
        
        super().__init__()
        self.network_evolution = network_evolution
        self.state_transition = state_transition
        self.confidence_weight = confidence_weight
        self.variational_posterior = variational_posterior
        self.register_buffer('rho_1', torch.tensor(rho_1, dtype=torch.float32))
        self.register_buffer('rho_2', torch.tensor(rho_2, dtype=torch.float32))

    
    def compute_state_likelihood_batch(self,
                                features: torch.Tensor,
                                states: torch.Tensor,
                                distances: torch.Tensor,
                                node_batch: torch.Tensor,
                                network_data,
                                gumbel_samples: List[Dict[str, torch.Tensor]],
                                max_timestep: int,
                                current_epoch: int = 0) -> Tuple[torch.Tensor, int]:
        """
        Compute state dynamics likelihood using marginal-based Gumbel-Softmax samples.
        
        Returns:
            tuple: (total_likelihood, actual_prediction_count)
                - total_likelihood: averaged across samples
                - actual_prediction_count: averaged number of predictions made
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
                pos_weight = base_pos_weight
            elif current_epoch < 350:  # Bridging phase
                progress = (current_epoch - 300) / 50
                pos_weight = base_pos_weight * (1 - 0.4 * progress)
            else:  # Rollout-dominated phase
                pos_weight = base_pos_weight * 0.6
            
            return pos_weight, 1.0

        total_likelihood = 0.0
        total_predictions = 0  # Track actual number of predictions made
        num_samples = len(gumbel_samples)
        
        for sample_idx, current_samples in enumerate(gumbel_samples):
            self.state_transition.broken_links_history.clear()
            self.state_transition.clear_influence_tracking()
            sample_likelihood = 0.0
            sample_predictions = 0  # Predictions for this sample
            
            for t in range(max_timestep):
                for decision_k in range(3):
                    # Find undecided households in batch
                    batch_undecided = []
                    for node_idx in node_batch:
                        if states[node_idx, t, decision_k] == 0:
                            batch_undecided.append(node_idx)
                    
                    if len(batch_undecided) == 0:
                        continue
                    
                    # Count actual predictions made
                    sample_predictions += len(batch_undecided)
                    
                    batch_undecided_tensor = torch.tensor(batch_undecided, dtype=torch.long)
                    
                    # Compute activation probabilities
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

                    assert not torch.isnan(activation_probs).any(), \
                        f"State likelihood: activation_probs NaN at t={t}, decision_k={decision_k}"

                    
                    # Get actual outcomes
                    actual_outcomes = states[batch_undecided, t+1, decision_k]
                    pos_weight, neg_weight = compute_dynamic_class_weights(actual_outcomes, current_epoch)
                    
                    # # Compute margin loss
                    # margin_penalty = self.margin_loss.compute_loss(activation_probs, actual_outcomes)
                    # weighted_likelihood = -pos_weight * margin_penalty
                    # sample_likelihood += weighted_likelihood

                    targets = actual_outcomes.float()
                    loss = adaptive_focal_loss(
                        activation_probs, 
                        targets, 
                        t, 
                        max_timestep,
                        base_alpha=0.85, 
                        base_gamma=2.5
                    )
                    sample_likelihood += (-loss)

            
            total_likelihood += sample_likelihood
            total_predictions += sample_predictions
        
        # Average across samples (num_samples must be divided)
        return total_likelihood / num_samples, total_predictions / num_samples
    

    def compute_rollout_state_likelihood_batch(self,
                                        features: torch.Tensor,
                                        states: torch.Tensor,
                                        distances: torch.Tensor,
                                        node_batch: torch.Tensor,
                                        network_data,
                                        gumbel_samples: List[Dict[str, torch.Tensor]],
                                        max_timestep: int,
                                        current_epoch: int,
                                        rollout_steps: int = 4,
                                        use_pred_prob: float = 0.5) -> Tuple[torch.Tensor, int]:
        """
        Rollout training: multi-step forward with scheduled sampling
        
        Args:
            rollout_steps: Number of forward steps to roll out (will be dynamically adjusted based on epoch)
            use_pred_prob: Probability of using model prediction vs ground truth (overridden by schedule)
            
        Returns:
            tuple: (total_likelihood, actual_prediction_count)
        """
        if not gumbel_samples or rollout_steps <= 0:
            return torch.tensor(0.0, device=features.device), 0

        device = features.device
        total_likelihood = 0.0
        total_predictions = 0  # Track actual number of predictions made
        num_samples = len(gumbel_samples)
        
        # 🔧 AGGRESSIVE ROLLOUT: Network frozen, can give strong signals immediately
        if current_epoch < 550:
            rollout_steps = 4  # Not used before epoch 550
            use_pred_prob = 0.0
        else:
            # From epoch 550: immediately use challenging settings
            rollout_steps = 8  # Long-term planning from the start
            # Aggressive scheduled sampling: quickly shift to using predictions
            if current_epoch < 600:
                # Fast ramp: 0.0 → 0.5 in 50 epochs
                progress = (current_epoch - 550) / 50
                use_pred_prob = 0.5 * progress
            else:
                # Continue to high value: 0.5 → 0.9
                progress = min(1.0, (current_epoch - 600) / 100)
                use_pred_prob = 0.5 + 0.4 * progress

        for sample_idx, current_samples in enumerate(gumbel_samples):
            self.state_transition.broken_links_history.clear()
            self.state_transition.clear_influence_tracking()
            sample_likelihood = 0.0
            sample_predictions = 0  # Predictions for this sample
            
            # Select rollout starting time points
            max_start_time = max_timestep - rollout_steps
            if max_start_time <= 0:
                continue
                
            for start_t in range(0, max_start_time, rollout_steps):  # Start every rollout_steps
                # Initialize: use ground truth states as starting point
                current_rollout_states = states[:, start_t, :].clone()
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
                        
                        # Count actual predictions made
                        sample_predictions += len(batch_undecided)
                        
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

                        assert not torch.isnan(activation_probs).any(), \
                            f"Rollout likelihood: activation_probs NaN at t={actual_t}, decision_k={decision_k}"

                        
                        # Get ground truth labels
                        true_outcomes = states[batch_undecided, actual_t + 1, decision_k]
                        
                        # Dynamic class weights with epoch-based adjustment
                        n_positive = true_outcomes.sum()
                        n_total = len(true_outcomes)
                        if n_positive > 0:
                            base_pos_weight = min((n_total - n_positive) / n_positive, 5.0)
                            
                            # Adjust penalty based on current epoch
                            if current_epoch < 300:  # Network learning phase
                                pos_weight = base_pos_weight
                            elif current_epoch < 350:  # Bridging phase
                                progress = (current_epoch - 300) / 50
                                pos_weight = base_pos_weight * (1 - 0.4 * progress)
                            else:  # Rollout-dominated phase
                                pos_weight = base_pos_weight * 0.6
                        else:
                            pos_weight = 1.0
                        
                        # Compute loss
                        loss = adaptive_focal_loss(
                            activation_probs, 
                            true_outcomes, 
                            actual_t, 
                            max_timestep,
                            base_alpha=0.85, 
                            base_gamma=2.5
                        )
                        step_likelihood += (-loss)
                        
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
            total_predictions += sample_predictions

        # Average across samples
        rollout_avg = total_likelihood / num_samples
        avg_predictions = total_predictions / num_samples
        print(f"📊 Rollout - Steps: {rollout_steps}, use_pred_prob: {use_pred_prob:.2f}, "
              f"likelihood: {rollout_avg.item():.4f}, predictions: {avg_predictions:.1f}")
        return rollout_avg, avg_predictions
      

    def compute_network_observation_likelihood_batch(self,
                                                marginal_probs: Dict[str, torch.Tensor],
                                                node_batch: torch.Tensor,
                                                network_data,
                                                max_timestep: int) -> Tuple[torch.Tensor, int]:
        """
        Network observation likelihood for ALL pairs (observed + hidden).
        
        For observed pairs: penalize if marginal doesn't match observation.
        For hidden pairs: use expected missing probability.
        """
        total_likelihood = torch.tensor(0.0, device=node_batch.device)
        pair_timestep_count = torch.tensor(0.0, device=node_batch.device)
        batch_nodes_set = set(node_batch.tolist())
        
        # Group pairs by timestep once
        pairs_by_time = {}
        for pair_key, marginal_prob in marginal_probs.items():
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
            if i not in batch_nodes_set and j not in batch_nodes_set:
                continue
            if t not in pairs_by_time:
                pairs_by_time[t] = []
            pairs_by_time[t].append((i, j, pair_key, marginal_prob))
        
        # Process each timestep
        for t in range(max_timestep + 1):
            for i, j, pair_key, marginal_prob in pairs_by_time.get(t, []):
                pair_timestep_count += 1
                
                if network_data.is_observed(i, j, t):
                    # Observed pair: reward correct type, penalize wrong types
                    observed_type = network_data.get_link_type(i, j, t)  # 1 or 2
                    
                    if observed_type == 1:
                        # Observed bonding: p(obs|bonding) = 1-ρ1, p(obs|others) ≈ 0
                        likelihood = (marginal_prob[1] * torch.log(1 - self.rho_1 + 1e-8) +
                                    marginal_prob[0] * torch.log(torch.tensor(1e-8)) +
                                    marginal_prob[2] * torch.log(torch.tensor(1e-8)))
                    else:  # observed_type == 2
                        # Observed bridging: p(obs|bridging) = 1-ρ2, p(obs|others) ≈ 0
                        likelihood = (marginal_prob[2] * torch.log(1 - self.rho_2 + 1e-8) +
                                    marginal_prob[0] * torch.log(torch.tensor(1e-8)) +
                                    marginal_prob[1] * torch.log(torch.tensor(1e-8)))
                    
                    total_likelihood += likelihood
                    
                else:
                    # Hidden pair: expected missing probability (same as before)
                    no_link_obs_prob = 1-1e-8  # Small value to avoid log(0)
                    missing_prob = (marginal_prob[0] * torch.log(torch.tensor(no_link_obs_prob)) +
                                marginal_prob[1] * torch.log(self.rho_1 + 1e-8) +
                                marginal_prob[2] * torch.log(self.rho_2 + 1e-8))
                    total_likelihood += missing_prob
        # try:
        #     # gradient through one observed marginal_prob
        #     g = torch.autograd.grad(total_likelihood, marginal_prob, retain_graph=True, allow_unused=True)
        #     if g[0] is not None:
        #         print("[ObsDiagGrad] observed grad norm (single):", g[0].norm().item())
        # except Exception as e:
        #     print("[ObsDiagGrad] error:", e)
        
        return total_likelihood, pair_timestep_count



    def compute_prior_likelihood_batch(self,
                                    conditional_probs: Dict[str, torch.Tensor],
                                    marginal_probs: Dict[str, torch.Tensor],
                                    features: torch.Tensor,
                                    states: torch.Tensor,
                                    distances: torch.Tensor,
                                    node_batch: torch.Tensor,
                                    network_data,
                                    max_timestep: int) -> Tuple[torch.Tensor, int]:
        """
        Vectorized prior likelihood computation for ALL pairs.
        
        This is a vectorized version that replaces nested Python loops with batched
        tensor operations for ~10-20x speedup while maintaining exact mathematical equivalence.
        """
        batch_nodes_set = set(node_batch.tolist())
        
        # Group pairs by timestep
        pairs_by_time = {}
        for pair_key in marginal_probs.keys():
            parts = pair_key.split('_')
            if len(parts) == 3:
                i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
                if i in batch_nodes_set or j in batch_nodes_set:
                    if t not in pairs_by_time:
                        pairs_by_time[t] = []
                    pairs_by_time[t].append((i, j))
        
        # ==================== INITIAL PRIOR (t=0) ====================
        t0_likelihoods = []
        t0_count = 0
        
        for i, j in pairs_by_time.get(0, []):
            pair_key = f"{i}_{j}_0"
            if pair_key not in marginal_probs:
                continue
            
            t0_count += 1
            π_ij = marginal_probs[pair_key]  # [3]
            
            # Get initial probabilities from network evolution model
            feat_i = features[i].unsqueeze(0)
            feat_j = features[j].unsqueeze(0)
            dist = distances[i, j].unsqueeze(0).unsqueeze(1)
            init_probs = self.network_evolution.initial_probabilities(feat_i, feat_j, dist).squeeze(0)  # [3]
            
            # Likelihood: Σ π(k) × log P_prior(k)
            likelihood = torch.sum(π_ij * torch.log(init_probs + 1e-8))
            t0_likelihoods.append(likelihood)
        
        # Sum all t=0 likelihoods
        if t0_likelihoods:
            t0_total = torch.stack(t0_likelihoods).sum()
        else:
            sample_tensor = next(iter(marginal_probs.values())) if marginal_probs else None
            if sample_tensor is not None:
                t0_total = torch.tensor(0.0, device=sample_tensor.device, dtype=sample_tensor.dtype)
            else:
                t0_total = torch.tensor(0.0)
        
        # ==================== TEMPORAL TRANSITIONS (t >= 1) ====================
        temporal_likelihoods = []
        temporal_count = 0
        
        for t in range(1, max_timestep + 1):
            # Collect all valid pairs at this timestep
            timestep_pairs = []  # List of (i, j, π_prev, π_cond, trans)
            
            for i, j in pairs_by_time.get(t, []):
                pair_key_curr = f"{i}_{j}_{t}"
                pair_key_prev = f"{i}_{j}_{t-1}"
                
                if pair_key_curr not in marginal_probs or pair_key_prev not in marginal_probs:
                    continue
                if pair_key_curr not in conditional_probs:
                    continue
                
                π_prev = marginal_probs[pair_key_prev]  # [3]
                π_cond = conditional_probs[pair_key_curr]  # [3, 3]
                
                # Get static transition matrix from prior
                trans = self.network_evolution.get_static_transition_probs(
                    states[i, t], states[j, t]
                )  # [3, 3]
                
                timestep_pairs.append((i, j, π_prev, π_cond, trans))
            
            if not timestep_pairs:
                continue
            
            # ========== VECTORIZED COMPUTATION FOR ALL PAIRS AT THIS TIMESTEP ==========
            # Stack all π_prev, π_cond, and trans matrices
            π_prev_batch = torch.stack([pair[2] for pair in timestep_pairs])  # [N_pairs, 3]
            π_cond_batch = torch.stack([pair[3] for pair in timestep_pairs])  # [N_pairs, 3, 3]
            trans_batch = torch.stack([pair[4] for pair in timestep_pairs])  # [N_pairs, 3, 3]
            
            # Compute joint posterior probabilities: π(k', k) = π(k') × π(k | k')
            # Broadcasting: [N_pairs, 3, 1] × [N_pairs, 3, 3] = [N_pairs, 3, 3]
            joint_probs = π_prev_batch.unsqueeze(-1) * π_cond_batch
            
            # Compute log of prior transition probabilities (with numerical stability)
            log_trans = torch.log(trans_batch + 1e-8)
            
            # Prior likelihood: Σ_k' Σ_k [π(k') × π(k|k')] × log P_prior(k'→k)
            # Sum over k_prev (dim=1) and k_curr (dim=2), keep batch dimension
            pair_likelihoods = torch.sum(joint_probs * log_trans, dim=(1, 2))  # [N_pairs]
            
            temporal_likelihoods.append(pair_likelihoods.sum())
            temporal_count += len(timestep_pairs)
        
        # Sum all temporal likelihoods
        if temporal_likelihoods:
            temporal_total = torch.stack(temporal_likelihoods).sum()
        else:
            sample_tensor = next(iter(conditional_probs.values())) if conditional_probs else None
            if sample_tensor is not None:
                temporal_total = torch.tensor(0.0, device=sample_tensor.device, dtype=sample_tensor.dtype)
            else:
                temporal_total = torch.tensor(0.0)
        
        total_likelihood = t0_total + temporal_total
        total_count = t0_count + temporal_count
        
        return total_likelihood, total_count


    
    def compute_posterior_entropy_batch(self, 
                                    conditional_probs: Dict[str, torch.Tensor],
                                    marginal_probs: Dict[str, torch.Tensor],
                                    network_data,
                                    node_batch: torch.Tensor,
                                    max_timestep: int) -> Tuple[torch.Tensor, int]:
        """
        Vectorized posterior entropy computation for ALL pairs (observed + hidden).
        
        This is a vectorized version that replaces nested Python loops with batched
        tensor operations for ~10-20x speedup while maintaining exact mathematical equivalence.
        
        Even observed pairs have entropy initially, which decreases as the model learns
        to give them near-deterministic posteriors through observation likelihood.
        """
        batch_nodes_set = set(node_batch.tolist())
        
        # ==================== INITIAL ENTROPY (t=0) ====================
        # Collect all t=0 marginals for pairs involving batch nodes
        t0_entropies = []
        t0_count = 0
        
        for pair_key, π_ij in marginal_probs.items():
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
            
            if t != 0:
                continue
            if i not in batch_nodes_set and j not in batch_nodes_set:
                continue
            
            # H = -Σ π(k) log π(k)
            entropy = -torch.sum(π_ij * torch.log(π_ij + 1e-8))
            t0_entropies.append(entropy)
            t0_count += 1
        
        # Sum all t=0 entropies
        if t0_entropies:
            t0_total = torch.stack(t0_entropies).sum()
        else:
            # Use same device/dtype as marginal_probs if available
            sample_tensor = next(iter(marginal_probs.values())) if marginal_probs else None
            if sample_tensor is not None:
                t0_total = torch.tensor(0.0, device=sample_tensor.device, dtype=sample_tensor.dtype)
            else:
                t0_total = torch.tensor(0.0)
        
        
        # ==================== TEMPORAL ENTROPY (t >= 1) ====================
        temporal_entropies = []
        temporal_count = 0
        
        for t in range(1, max_timestep + 1):
            # Collect all pairs at this timestep
            timestep_pairs = []  # List of (pair_key, π_prev, π_conditional)
            
            for pair_key, π_conditional in conditional_probs.items():
                parts = pair_key.split('_')
                if len(parts) != 3:
                    continue
                i, j, time = int(parts[0]), int(parts[1]), int(parts[2])
                
                if time != t:
                    continue
                if i not in batch_nodes_set and j not in batch_nodes_set:
                    continue
                
                pair_key_prev = f"{i}_{j}_{t-1}"
                if pair_key_prev not in marginal_probs:
                    continue
                
                π_prev = marginal_probs[pair_key_prev]  # [3]
                timestep_pairs.append((pair_key, π_prev, π_conditional))
            
            if not timestep_pairs:
                continue
            
            # ========== VECTORIZED COMPUTATION FOR ALL PAIRS AT THIS TIMESTEP ==========
            # Stack all π_prev and π_conditional for batch processing
            π_prev_batch = torch.stack([pair[1] for pair in timestep_pairs])  # [N_pairs, 3]
            π_cond_batch = torch.stack([pair[2] for pair in timestep_pairs])  # [N_pairs, 3, 3]
            
            # Compute joint probabilities: π(k', k) = π(k') × π(k | k')
            # Broadcasting: [N_pairs, 3, 1] × [N_pairs, 3, 3] = [N_pairs, 3, 3]
            joint_probs = π_prev_batch.unsqueeze(-1) * π_cond_batch
            
            # Compute log of conditional probabilities (with numerical stability)
            log_conditionals = torch.log(π_cond_batch + 1e-8)
            
            # Conditional entropy: H = -Σ_k' Σ_k [π(k') × π(k|k')] × log π(k|k')
            # Sum over k_prev (dim=1) and k_curr (dim=2), keep batch dimension
            pair_entropies = -torch.sum(joint_probs * log_conditionals, dim=(1, 2))  # [N_pairs]
            
            temporal_entropies.append(pair_entropies.sum())
            temporal_count += len(timestep_pairs)
        
        # Sum all temporal entropies
        if temporal_entropies:
            temporal_total = torch.stack(temporal_entropies).sum()
        else:
            # Use same device/dtype as conditional_probs if available
            sample_tensor = next(iter(conditional_probs.values())) if conditional_probs else None
            if sample_tensor is not None:
                temporal_total = torch.tensor(0.0, device=sample_tensor.device, dtype=sample_tensor.dtype)
            else:
                temporal_total = torch.tensor(0.0)
        
        total_entropy = t0_total + temporal_total
        total_count = t0_count + temporal_count
        
        return total_entropy, total_count
    

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
    



    def compute_type_specific_density_penalty(self, marginal_probs, network_data, max_timestep,
                                          temperature=0.01, balance_factor=1.0,
                                          penalty_strength=1.0,
                                          rho1_scale=1.0, rho2_scale=1.0,
                                          current_epoch: int = 0):
        """
        Temporal-aware soft density penalty
        -----------------------------------
        • Uses known random-miss rates (ρ₁, ρ₂) to infer expected hidden counts
        within the candidate subset at each timestep.
        • Applies a dynamic softmax temperature schedule (1.0 → 0.7 → 0.5)
        for smoother yet discriminative gradients.
        • Uses a log-ratio² penalty between predicted and expected counts.
        """

        import torch
        import torch.nn.functional as F

        eps = 1e-8
        device = next((v.device for v in marginal_probs.values()), torch.device("cpu")) \
                if len(marginal_probs) > 0 else torch.device("cpu")

        neighbor_index = getattr(network_data, "neighbor_index", None)

        def in_cand(i, j):
            """Check whether (i, j) lies within the candidate neighbor subset."""
            if neighbor_index is None:
                return True
            return (j in neighbor_index[i]) or (i in neighbor_index[j])

        # --------------------------------------------------------
        # 1) Group marginal probabilities by timestep
        # --------------------------------------------------------
        timestep_probs = {}
        for key, pij in marginal_probs.items():
            parts = key.split('_')
            if len(parts) != 3:
                continue
            t = int(parts[2])
            timestep_probs.setdefault(t, []).append(pij)

        total_penalty = torch.tensor(0.0, device=device)

        # Logging accumulators
        tot_exp_hidden_bond = 0.0
        tot_exp_hidden_brid = 0.0
        tot_pred_bond = 0.0
        tot_pred_brid = 0.0
        tot_cand_pairs = 0

        # --------------------------------------------------------
        # 2) Compute effective missing rates within candidate set
        # --------------------------------------------------------
        r1 = max(0.0, min(0.999, float(self.rho_1) * float(rho1_scale)))
        r2 = max(0.0, min(0.999, float(self.rho_2) * float(rho2_scale)))
        r1_tensor = torch.tensor(r1, device=device)
        r2_tensor = torch.tensor(r2, device=device)

        # --------------------------------------------------------
        # 3) Dynamic temperature schedule based on epoch
        # --------------------------------------------------------
        if current_epoch < 200:
            temp_eff = 1.0
        elif current_epoch < 400:
            temp_eff = 0.7
        else:
            temp_eff = 0.5

        # --------------------------------------------------------
        # 4) Iterate through each timestep
        # --------------------------------------------------------
        steps_count = 0
        for t in range(max_timestep + 1):

            # ---- (a) Count observed positive edges in candidate set
            m1, m2 = 0, 0
            observed_edges_t = network_data.get_observed_edges_at_time(t)
            for (i, j, link_type) in observed_edges_t:
                if in_cand(i, j):
                    if link_type == 1: m1 += 1
                    elif link_type == 2: m2 += 1

            m1_t = torch.tensor(float(m1), device=device)
            m2_t = torch.tensor(float(m2), device=device)

            # ---- (b) Expected hidden counts inferred from missing rates
            exp_hidden_bond_t = m1_t / (1.0 - r1_tensor + eps)
            if t == 0:
                exp_hidden_brid_t = m2_t / (1.0 - r2_tensor + eps)
            else:
                # Later timesteps: mainly persistence; can decay if needed
                decay_factor = 1.0
                exp_hidden_brid_t = (m2_t / (1.0 - r2_tensor + eps)) * decay_factor

            # ---- (c) Predicted counts using softmax(log(p) / τ)
            pred_bond_t = torch.tensor(0.0, device=device)
            pred_brid_t = torch.tensor(0.0, device=device)
            n_cand_t = 0

            if t in timestep_probs and len(timestep_probs[t]) > 0:
                prob_stack = torch.stack(timestep_probs[t]).to(device)  # [num_pairs_t, 3]
                n_cand_t = prob_stack.shape[0]

                # Convert probabilities to logits for stable softmax reweighting
                logits = torch.log(prob_stack + eps)
                soft_probs = F.softmax(logits / temp_eff, dim=1)

                pred_bond_t = soft_probs[:, 1].sum()
                pred_brid_t = soft_probs[:, 2].sum()

            # ---- (d) Compute smooth log-ratio² penalties
            bond_pen_t = torch.tensor(0.0, device=device)
            if exp_hidden_bond_t.item() > 1e-6:
                ratio_b = (pred_bond_t + 1e-8) / (exp_hidden_bond_t + 1e-8)
                bond_pen_t = (torch.log(ratio_b).abs()) ** 2
                bond_pen_t = balance_factor * bond_pen_t

            brid_pen_t = torch.tensor(0.0, device=device)
            if exp_hidden_brid_t.item() > 1e-6:
                ratio_r = (pred_brid_t + 1e-8) / (exp_hidden_brid_t + 1e-8)
                brid_pen_t = (torch.log(ratio_r).abs()) ** 2

            # ---- (e) Time weighting (uniform by default)
            time_weight = 1.0

            total_penalty += time_weight * (bond_pen_t + brid_pen_t)
            steps_count += 1

            # ---- (f) Logging accumulators
            tot_exp_hidden_bond += float(exp_hidden_bond_t.item())
            tot_exp_hidden_brid += float(exp_hidden_brid_t.item())
            tot_pred_bond += float(pred_bond_t.item())
            tot_pred_brid += float(pred_brid_t.item())
            tot_cand_pairs += n_cand_t

        # --------------------------------------------------------
        # 5) Normalize and scale
        # --------------------------------------------------------
        if steps_count == 0:
            steps_count = 1
        final_penalty = penalty_strength * (total_penalty / float(steps_count))

        # --------------------------------------------------------
        # 6) Verbose logging (same format as your original)
        # --------------------------------------------------------
        if neighbor_index is not None and max_timestep >= 0:
            print(f"[SparsePenalty] avg candidate pairs per t ≈ {tot_cand_pairs / float(max_timestep + 1):.1f}")
        print(f"  Expected hidden (candidates): bonding={tot_exp_hidden_bond:.1f}, bridging={tot_exp_hidden_brid:.1f}")
        print(f"  Predicted (candidates):       bonding={tot_pred_bond:.1f}, bridging={tot_pred_brid:.1f}")
        print(f"  Final penalty: {float(final_penalty.item()):.3f}")

        return final_penalty

    
    def compute_sparsity_penalty(self, marginal_probs, sparsity_strength=5.0):
        """
        General sparsity prior: penalize predicting links.
        """
        total_prior = 0.0
        count = 0
        
        for pair_key, marginal in marginal_probs.items():
            prior = (marginal[0] * torch.log(torch.tensor(0.99)) +
                    (marginal[1] + marginal[2]) * torch.log(torch.tensor(0.01)))
            total_prior += prior
            count += 1
        
        # Normalize by number of pairs
        if count > 0:
            total_prior = total_prior / count
        
        # Apply strength
        return -sparsity_strength * total_prior


    
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
                        lambda_constraint: float = 0.00,
                        current_epoch: int = 0) -> Dict[str, torch.Tensor]:        
        """
        Complete ELBO computation with per-example normalization and component weighting.
        """

        def get_dynamic_weights(epoch):
            if epoch < 150:  
                return {
                    'state': 0.0, 
                    'rollout': 0.0,
                    'observation': 10.0, 
                    'prior': 1.0,                       
                    'entropy': 1.0, 
                    'sparse_penalty': 0.0,    
                    'density_penalty': 1.0,    
                }
            elif epoch < 350:
                progress = (epoch - 150) / 200
                state_weight = 0.1  
                prior_weight = 1.0 + 9.0 * progress 
                observation_weight = 10.0 + 10.0 * progress
                # 🔧 FIX: Gradual density weight increase (1.0 → 10.0)
                # Smoother than jumping directly to 10
                density_weight = 1.0 + 9.0 * progress  # 1 → 10 over 200 epochs
                return {
                    'state': state_weight, 
                    'rollout': 0.0,
                    'observation': observation_weight, 
                    'prior': prior_weight,
                    'entropy': 1.0, 
                    'sparse_penalty': 0.0,
                    'density_penalty': density_weight  # Was fixed 1.0, now 1→10
                }
            elif epoch < 550:
                return {
                    'state': 0.1, 
                    'rollout': 0.0,
                    'observation': 20.0, 
                    'prior': 10.0,    
                    'entropy': 1.0, 
                    'sparse_penalty': 0.0,
                    'density_penalty': 10.0  # 🔧 FIX: Keep at 10 (half of obs)
                }
            else:  # epoch >= 550: Prediction learning phase
                # 🔧 STRONG SIGNAL: Network is frozen, give strong signals immediately
                # No need for gradual ramp-up since network structure won't be affected
                return {
                    'state': 10000.0,                    # Strong signal for one-step prediction
                    'rollout': 100.0,                  # Strong signal for multi-step rollout
                    'observation': 0.0, 
                    'prior': 0.0,
                    'entropy': 0.0, 
                    'sparse_penalty': 0.0,         
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
        if current_epoch < 550:
            state_likelihood_raw, state_count = torch.tensor(0.0), 1
            rollout_likelihood_raw, rollout_count = torch.tensor(0.0), 1
        else:
            state_likelihood_raw, state_count = self.compute_state_likelihood_batch(
            features, states, distances, node_batch, network_data, gumbel_samples, max_timestep, current_epoch=current_epoch
                    )
        timing_results['state_likelihood'] = time.time() - start_time

        start_time = time.time()
        if current_epoch < 550:
            rollout_likelihood_raw, rollout_count = torch.tensor(0.0), 0
        else:
            # 🔧 IMPROVED: Remove hardcoded parameters, let function decide based on epoch
            rollout_likelihood_raw, rollout_count = self.compute_rollout_state_likelihood_batch(
                features, states, distances, node_batch, network_data, gumbel_samples, 
                max_timestep, current_epoch=current_epoch
                # rollout_steps and use_pred_prob are now dynamically determined inside the function
            )
        timing_results['rollout_likelihood'] = time.time() - start_time

        start_time = time.time()
        observation_likelihood_raw, obs_count = self.compute_network_observation_likelihood_batch(
        marginal_probs, node_batch, network_data, max_timestep
            )
        timing_results['observation_likelihood'] = time.time() - start_time

        start_time = time.time()
        prior_likelihood_raw, prior_count = self.compute_prior_likelihood_batch(
        conditional_probs, marginal_probs, features, states, distances, 
        node_batch, network_data, max_timestep
        )
        timing_results['prior_likelihood'] = time.time() - start_time
        print("prior count",prior_count)

        # Compute other components
        start_time = time.time()
        posterior_entropy_raw, entropy_count = self.compute_posterior_entropy_batch(
        conditional_probs, marginal_probs, network_data, node_batch, max_timestep
        )
        timing_results['posterior_entropy'] = time.time() - start_time

        start_time = time.time()
        # sparsity_penalty = self.compute_sparsity_penalty(marginal_probs, network_data, node_batch, max_timestep)
        sparsity_penalty = self.compute_sparsity_penalty(marginal_probs, sparsity_strength=3.0)
        timing_results['sparsity_penalty'] = time.time() - start_time

        start_time = time.time()
        timestep_density_penalty = self.compute_type_specific_density_penalty(
            marginal_probs, network_data, max_timestep
        )
        timing_results['timestep_density_penalty'] = time.time() - start_time


        print(f"Raw values - State: {state_likelihood_raw.item():.2f}, "
            f"Obs: {observation_likelihood_raw.item():.2f}, "
            f"Prior: {prior_likelihood_raw.item():.2f}, "
            f"Entropy: {posterior_entropy_raw.item():.2f},"
            f" Sparsity: {sparsity_penalty.item():.2f}, "
            f"Timestep Density Penalty: {timestep_density_penalty.item():.2f}")
        

        # Calculate actual counts for proper normalization
        # total_state_predictions =  max_timestep * 3  # 3 decision types
        total_state_predictions = len(node_batch) * max_timestep * 3  # 3 decision types per sample
        total_network_pairs = len(marginal_probs)  # Actual pairs being evaluated
        print(f"Total state predictions: {total_state_predictions}, Total network pairs: {total_network_pairs}")
        
        # Normalize by actual counts
        state_likelihood_per_prediction = state_likelihood_raw / state_count if state_count > 0 else torch.tensor(0.0)
        rollout_likelihood_per_prediction = rollout_likelihood_raw / rollout_count if rollout_count > 0 else torch.tensor(0.0)
        network_likelihood_per_pair = observation_likelihood_raw / obs_count if obs_count > 0 else torch.tensor(0.0)
        prior_likelihood_per_pair = prior_likelihood_raw / prior_count if prior_count > 0 else torch.tensor(0.0)
        entropy_per_pair = posterior_entropy_raw / entropy_count if entropy_count > 0 else torch.tensor(0.0)
        # confidence_per_pair = confidence_reg_raw / confidence_count if confidence_count > 0 else torch.tensor(0.0)
        
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
        weighted_density_penalty = weight['density_penalty'] * timestep_density_penalty
        weighted_sparsity_penalty = weight['sparse_penalty'] * sparsity_penalty

        
        # Total weighted ELBO
        total_elbo = (weighted_state_likelihood + weighted_rollout_likelihood + weighted_observation_likelihood +
                    weighted_prior_likelihood + weighted_posterior_entropy 
                     - weighted_density_penalty - weighted_sparsity_penalty)

        print(f"Weighted components - State: {weighted_state_likelihood.item():.4f}, "
            f"Rollout: {weighted_rollout_likelihood.item():.4f}, "
            f"Obs: {weighted_observation_likelihood.item():.4f}, "
            f"Prior: {weighted_prior_likelihood.item():.4f}, "
            f"Entropy: {weighted_posterior_entropy.item():.4f}")
        print( f"Timestep Density Penalty: {weighted_density_penalty.item():.4f}, "
            f"Sparsity Penalty: {weighted_sparsity_penalty.item():.4f}")
        print(f"Total weighted ELBO: {total_elbo.item():.4f}")

        print("\n=== FUNCTION TIMING ANALYSIS ===")
        total_time = sum(timing_results.values())
        for func_name, exec_time in timing_results.items():
            percentage = (exec_time / total_time) * 100 if total_time > 0 else 0
            print(f"{func_name:25s}: {exec_time:.4f}s ({percentage:5.1f}%)")
        print(f"{'TOTAL TIME':25s}: {total_time:.4f}s (100.0%)")


        # --- BEGIN: Gradient attribution for PREDICTION networks (InfluenceNN & SelfNN) ---
        print("\n=== PREDICTION GRADIENT ATTRIBUTION (State & Rollout Likelihood) ===")
        print("  This shows the GRADIENT NORM (L2 norm of all parameter gradients)")
        print("  for each NN from each loss term.")
        print("  Helps identify which NN is getting stronger training signals.\n")
        
        # Get parameters for prediction networks
        params_influence = [p for p in self.state_transition.influence_nn.parameters() if p.requires_grad]
        params_self = [p for p in self.state_transition.self_nn.parameters() if p.requires_grad]

        def safe_grad_norm_of(scalar, params, name="unknown"):
            """
            Compute gradient norm (L2 norm) safely.
            
            This computes: sqrt(sum(||∂Loss/∂θ||² for all parameters θ))
            i.e., the total magnitude of gradients flowing to this NN from this loss term.
            """
            if (scalar is None) or (not torch.is_tensor(scalar)) or (not scalar.requires_grad) or len(params) == 0:
                return 0.0
            try:
                # Compute gradients ∂scalar/∂params
                g = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
                # Sum squared norms: sum(||grad_i||²)
                total = 0.0
                for gi in g:
                    if gi is not None:
                        total += gi.norm().item() ** 2
                # Return L2 norm: sqrt(sum)
                return (total ** 0.5)
            except RuntimeError as e:
                print(f"  [GradDiag] Error computing grad for {name}: {e}")
                return 0.0

        # Compute gradient contributions for InfluenceNN
        grad_influence = {}
        grad_influence['state'] = safe_grad_norm_of(weighted_state_likelihood, params_influence, "Influence.state")
        grad_influence['rollout'] = safe_grad_norm_of(weighted_rollout_likelihood, params_influence, "Influence.rollout")
        
        # Compute gradient contributions for SelfNN
        grad_self = {}
        grad_self['state'] = safe_grad_norm_of(weighted_state_likelihood, params_self, "Self.state")
        grad_self['rollout'] = safe_grad_norm_of(weighted_rollout_likelihood, params_self, "Self.rollout")
        
        # Print results
        print(f"  [InfluenceNN] State: {grad_influence['state']:.4e}, Rollout: {grad_influence['rollout']:.4e}, "
              f"Total: {(grad_influence['state']**2 + grad_influence['rollout']**2)**0.5:.4e}")
        print(f"  [SelfNN]      State: {grad_self['state']:.4e}, Rollout: {grad_self['rollout']:.4e}, "
              f"Total: {(grad_self['state']**2 + grad_self['rollout']**2)**0.5:.4e}")
        
        # Compare relative magnitudes
        total_grad_influence = (grad_influence['state']**2 + grad_influence['rollout']**2)**0.5
        total_grad_self = (grad_self['state']**2 + grad_self['rollout']**2)**0.5
        
        if total_grad_influence + total_grad_self > 0:
            ratio_influence = total_grad_influence / (total_grad_influence + total_grad_self)
            ratio_self = total_grad_self / (total_grad_influence + total_grad_self)
            print(f"  Gradient Distribution: InfluenceNN={ratio_influence*100:.1f}%, SelfNN={ratio_self*100:.1f}%")
        
        print("=" * 50)
        # --- END ---


        return {
            'state_likelihood': weighted_state_likelihood,
            'rollout_likelihood': weighted_rollout_likelihood,
            'observation_likelihood': weighted_observation_likelihood, 
            'prior_likelihood': weighted_prior_likelihood,
            'posterior_entropy': weighted_posterior_entropy,
            'density_penalty': weighted_density_penalty,
            'sparse_penalty': weighted_sparsity_penalty,
            'total_elbo': total_elbo
        }