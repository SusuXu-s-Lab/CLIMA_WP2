import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple
from models import NetworkEvolution, StateTransition
from collections import defaultdict

def adaptive_focal_loss(p, y, timestep, total_timesteps, valid=None, base_alpha=0.60, base_gamma=2.0):
    """
    Vectorized adaptive focal loss with valid mask.
    
    CALIBRATED VERSION: Reduced alpha and gamma to prevent overconfident predictions.
    
    Args:
        p: [B, 3] or [B, T, 3] predicted hazard probabilities
        y: [B, 3] or [B, T, 3] event labels (0/1)
        valid: [B, 3] or [B, T, 3] mask (1 for valid/alive, 0 for ignore)
        timestep: int, current time step
        total_timesteps: int, total number of time steps
        base_alpha: Base value for class weight (REDUCED from 0.85 to 0.60)
        base_gamma: Base focusing parameter (REDUCED from 2.5 to 2.0)
        
    Returns:
        loss: tensor, per-sample loss (0 for invalid positions)
    """
    original_shape = y.shape
    if len(original_shape) == 3:
        # Flatten [B, T, 3] -> [B*T, 3]
        B, T, K = original_shape
        p = p.reshape(B * T, K)
        y = y.reshape(B * T, K)
        if valid is not None:
            valid = valid.reshape(B * T, K)
    else:
        B, K = original_shape
    
    eps = 1e-6
    p = torch.clamp(p, eps, 1 - eps)
    if valid is None:
        valid = torch.ones_like(y)
    
    # Compute positive/negative counts only on valid positions
    n_pos = (y * valid).sum(dim=0)  # [3]
    n_valid = valid.sum(dim=0)
    n_neg = n_valid - n_pos
    n_pos_safe = torch.where(n_pos == 0, torch.ones_like(n_pos), n_pos)
    ratio = n_neg / n_pos_safe
    print(f"[AdaptiveFocalLoss] timestep={timestep}, n_pos={n_pos.tolist()}, n_neg={n_neg.tolist()}, ratio={ratio.tolist()}")
    
    # CHANGE 1: Reduce alpha growth rate (0.1 -> 0.01)
    alpha = 1 - 1 / (1 + 0.01 * ratio)  # Slower growth
    alpha = torch.clamp(alpha, base_alpha, 0.75)  # REDUCED upper limit: 0.99 -> 0.75
    
    # CHANGE 2: Remove time_boost (was causing alpha to grow too high)
    # time_boost = 1 + 0.5 * (timestep / total_timesteps)  # REMOVED
    # alpha = torch.clamp(alpha * time_boost, base_alpha, 0.99)  # REMOVED
    
    # CHANGE 3: Reduce gamma growth (0.5 -> 0.2)
    gamma = base_gamma + 0.2 * torch.log10(torch.clamp(ratio, 1, 1000))  # Slower growth
    gamma = torch.clamp(gamma, 1.5, 2.5)  # Limit range
    
    # Broadcast alpha/gamma
    alpha = alpha.unsqueeze(0).expand_as(p)
    gamma = gamma.unsqueeze(0).expand_as(p)
    
    # Compute focal loss
    loss_pos = -alpha * (1 - p) ** gamma * y * torch.log(p)
    loss_neg = -(1 - alpha) * p ** gamma * (1 - y) * torch.log(1 - p)
    
    # For decision types with no positive, only compute negative loss
    mask_pos = (n_pos > 0).float().unsqueeze(0).expand_as(p)
    loss_pos = loss_pos * mask_pos
    
    # Only keep valid positions
    total_loss = (loss_pos + loss_neg) * valid
    
    if len(original_shape) == 3:
        total_loss = total_loss.reshape(B, T, K)
    
    return total_loss


class ELBOComputation(nn.Module):
    """ELBO computation using seq2seq state likelihood"""

    def __init__(self, network_evolution, state_transition,
                 rho_1: float = 0.5, rho_2: float = 0.5, 
                 confidence_weight: float = 0.0, variational_posterior=None):
        
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
        Compute state likelihood using seq2seq approach.
        
        Args:
            features: [N, feat_dim]
            states: [N, T+1, 3] - full state tensor
            distances: [N, N]
            node_batch: [batch_size] - household indices in this batch
            network_data: network structure
            gumbel_samples: List of sample dicts
            max_timestep: T
            
        Returns:
            loss: scalar loss
            n_valid: number of valid supervisions
        """
        device = states.device
        N, T_plus_1, K = states.shape
        T = max_timestep
        
        # Use first gumbel sample
        gumbel_sample = gumbel_samples[0] if isinstance(gumbel_samples, list) else gumbel_samples
        
        # Compute activation probabilities for all timesteps: [N, T, 3]
        p_seq = self.state_transition.compute_activation_probability_seq(
            features, states, distances, network_data, gumbel_sample
        )
        
        # Event indicator: s[t+1] - s[t] for t=0..T-1
        event = states[:, 1:T+1, :] - states[:, :T, :]  # [N, T, 3]
        event = event.float().clamp(0, 1)
        
        # Alive mask: only supervise at positions where not yet activated
        alive_mask = 1.0 - states[:, :T, :].float()  # [N, T, 3]
        
        # Compute focal loss for all timesteps (no time adaptation)
        loss_per_sample = adaptive_focal_loss(
            p_seq, event, timestep=0, total_timesteps=T, valid=alive_mask
        )  # [N, T, 3]
        
        # Sum over all valid positions
        total_loss = loss_per_sample.sum()
        n_valid = int(alive_mask.sum().item())
        
        # ============ NEW: Monitor prediction statistics ============
        # Only print every 10 epochs to avoid spam
        if current_epoch % 10 == 0:
            decision_names = ['vacant', 'repair', 'sell']
            print(f"\n[Prediction Stats - Epoch {current_epoch}]")
            print("="*70)
            
            # Per-timestep statistics
            for t in range(min(T, 15)):  # Only show first 15 timesteps to avoid clutter
                # For each decision type
                stats_line = f"t={t:2d}: "
                for k in range(3):
                    # Get valid predictions at this timestep
                    valid_mask = alive_mask[:, t, k] > 0
                    
                    if valid_mask.sum() > 0:
                        preds_t_k = p_seq[:, t, k][valid_mask]
                        events_t_k = event[:, t, k][valid_mask]
                        
                        mean_pred = preds_t_k.mean().item()
                        actual_rate = events_t_k.mean().item()
                        n_valid_tk = valid_mask.sum().item()
                        
                        stats_line += f"{decision_names[k]}: pred={mean_pred:.4f} (actual={actual_rate:.4f}, n={n_valid_tk:3d}) | "
                    else:
                        stats_line += f"{decision_names[k]}: - | "
                
                print(stats_line)
            
            # Overall statistics (across all timesteps)
            print("\n" + "-"*70)
            print("Overall Statistics (all timesteps):")
            for k, dec_name in enumerate(decision_names):
                valid_mask_all = alive_mask[:, :, k] > 0
                
                if valid_mask_all.sum() > 0:
                    preds_all = p_seq[:, :, k][valid_mask_all]
                    events_all = event[:, :, k][valid_mask_all]
                    
                    print(f"  {dec_name.upper()}:")
                    print(f"    Mean pred: {preds_all.mean():.4f} ± {preds_all.std():.4f}")
                    print(f"    Range: [{preds_all.min():.4f}, {preds_all.max():.4f}]")
                    print(f"    Actual event rate: {events_all.mean():.4f}")
                    print(f"    Valid samples: {valid_mask_all.sum().item()}")
            
            print("="*70 + "\n")
        # ============ END: Monitor prediction statistics ============
        
        return -total_loss, n_valid
    
    def compute_rollout_likelihood(self,
                              features: torch.Tensor,
                              initial_states: torch.Tensor,
                              distances: torch.Tensor,
                              network_data,
                              gumbel_samples: Dict[str, torch.Tensor],
                              max_timestep: int,
                              true_states: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Compute rollout likelihood by simulating forward without teacher forcing.
        
        Args:
            features: [N, feat_dim]
            initial_states: [N, 1, 3] - states at t=0
            distances: [N, N]
            network_data: network structure
            gumbel_samples: single sample dict
            max_timestep: T
            true_states: [N, T+1, 3] - ground truth for supervision
            
        Returns:
            loss: scalar loss
            n_valid: number of valid supervisions
        """
        device = features.device
        N = features.shape[0]
        T = max_timestep
        
        # Initialize simulated states list (avoid inplace operations)
        simulated_states_list = [initial_states.squeeze(1)]  # List of [N, 3] tensors
        
        total_loss = 0.0
        n_valid = 0
        
        for t in range(T):
            # Build states tensor from list
            current_states = torch.stack(simulated_states_list, dim=1)  # [N, t+1, 3]
            
            # Pad to match expected shape [N, t+2, 3] for seq method
            temp_states = torch.cat([
                current_states,
                torch.zeros((N, 1, 3), device=device)
            ], dim=1)  # [N, t+2, 3]
            
            # Compute activation probability for timestep t
            p_seq_partial = self.state_transition.compute_activation_probability_seq(
                features, temp_states, distances, network_data, gumbel_samples
            )  # [N, t+1, 3]
            
            p_t = p_seq_partial[:, -1, :]  # [N, 3] - probability for t+1
            
            # Sample binary outcomes (detach to prevent backprop through sampling)
            sampled_events = torch.bernoulli(p_t.detach())  # [N, 3]
            
            # Update states: once activated, stay activated
            new_state = torch.maximum(simulated_states_list[-1], sampled_events)
            simulated_states_list.append(new_state)
            
            # Compute loss for this timestep
            event_true = true_states[:, t+1, :] - true_states[:, t, :]
            event_true = event_true.float().clamp(0, 1)
            alive_mask = 1.0 - true_states[:, t, :].float()
            
            loss_t = adaptive_focal_loss(
                p_t.unsqueeze(1), event_true.unsqueeze(1), 
                timestep=0, total_timesteps=T, valid=alive_mask.unsqueeze(1)
            ).sum()
            
            total_loss += loss_t
            n_valid += int(alive_mask.sum().item())
    
        return -total_loss, n_valid
      

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
    


    # def compute_type_specific_density_penalty(self, marginal_probs, network_data, max_timestep,
    #                                         temperature=0.01, balance_factor=1.0,
    #                                         penalty_strength=1.0,
    #                                         rho1_scale=1.0, rho2_scale=1.0):
    #     """
    #     Temporal-aware sparse density penalty:
    #     - Adjusts expected hidden counts based on timestep
    #     - t=0: Higher expectation (all links form here)
    #     - t>0: Lower expectation (only persistence, no new formation)
    #     """
    #     import torch
    #     import torch.nn.functional as F

    #     device = next((v.device for v in marginal_probs.values()), torch.device("cpu")) \
    #             if len(marginal_probs) > 0 else torch.device("cpu")
    #     eps = 1e-8

    #     neighbor_index = getattr(network_data, "neighbor_index", None)
    #     def in_cand(i, j):
    #         if neighbor_index is None:
    #             return True
    #         return (j in neighbor_index[i]) or (i in neighbor_index[j])

    #     # group candidate marginals by timestep
    #     timestep_probs = {}
    #     for key, pij in marginal_probs.items():
    #         parts = key.split('_')
    #         if len(parts) != 3: continue
    #         t = int(parts[2])
    #         timestep_probs.setdefault(t, []).append(pij)

    #     total_penalty = torch.tensor(0.0, device=device)

    #     # logs (for visibility)
    #     tot_exp_hidden_bond = 0.0
    #     tot_exp_hidden_brid = 0.0
    #     tot_pred_bond = 0.0
    #     tot_pred_brid = 0.0
    #     tot_cand_pairs = 0

    #     # candidate-effective missing rates (optionally scaled)
    #     r1 = max(0.0, min(0.999, float(self.rho_1) * float(rho1_scale)))
    #     r2 = max(0.0, min(0.999, float(self.rho_2) * float(rho2_scale)))
    #     r1_tensor = torch.tensor(r1, device=device)
    #     r2_tensor = torch.tensor(r2, device=device)

    #     for t in range(max_timestep + 1):
    #         # 1) count observed positives in candidates at t
    #         m1 = 0  # observed bonding inside candidates
    #         m2 = 0  # observed bridging inside candidates
    #         observed_edges_t = network_data.get_observed_edges_at_time(t)
    #         for (i, j, link_type) in observed_edges_t:
    #             if in_cand(i, j):
    #                 if link_type == 1: m1 += 1
    #                 elif link_type == 2: m2 += 1

    #         m1_t = torch.tensor(float(m1), device=device)
    #         m2_t = torch.tensor(float(m2), device=device)

    #         # 2) MODIFIED: Temporal-aware expected hidden counts
    #         # Bonding links: stable across time (same expectation)
    #         exp_hidden_bond_t = m1_t / (1.0 - r1_tensor + eps)
            
    #         # Bridging links: temporal adjustment
    #         if t == 0:
    #             # t=0: Standard expectation (formation timestep)
    #             exp_hidden_brid_t = m2_t / (1.0 - r2_tensor + eps)
    #         else:
    #             # t>0: Lower expectation (only persistence, no new formation)
    #             # Simple decay model - you can adjust this factor
    #             decay_factor = 1.0  # Exponential decay
    #             exp_hidden_brid_t = m2_t / (1.0 - r2_tensor + eps) * decay_factor

    #         # 3) model predicted counts on candidates at t (unchanged)
    #         pred_bond_t = torch.tensor(0.0, device=device)
    #         pred_brid_t = torch.tensor(0.0, device=device)
    #         n_cand_t = 0
    #         if t in timestep_probs and len(timestep_probs[t]) > 0:
    #             prob_stack = torch.stack(timestep_probs[t]).to(device)
    #             n_cand_t = prob_stack.shape[0]
    #             logits = torch.log(prob_stack + eps)
    #             sharp = F.softmax(logits / temperature, dim=1)
    #             pred_bond_t = sharp[:, 1].sum()
    #             pred_brid_t = sharp[:, 2].sum()

    #         # 4) penalties with temporal weighting
    #         bond_pen_t = torch.tensor(0.0, device=device)
    #         if exp_hidden_bond_t.item() > 0.1:
    #             rel_err = torch.abs(pred_bond_t - exp_hidden_bond_t) / (exp_hidden_bond_t + eps)
    #             bond_pen_t = balance_factor * rel_err
    #         bond_pen_t = 5.0 * torch.clamp(bond_pen_t, 0.0, 5.0)

    #         brid_pen_t = torch.tensor(0.0, device=device)
    #         if exp_hidden_brid_t.item() > 0.1:
    #             ratio = (pred_brid_t + 1.0) / (exp_hidden_brid_t + 1.0)
    #             brid_pen_t = torch.abs(torch.log(ratio))
    #             # brid_pen_t = torch.abs(ratio)
    #         brid_pen_t = 5 * torch.clamp(brid_pen_t, 0.0, 5.0)

    #         # bond_pen_t = bond_pen_t * (brid_pen_t/(brid_pen_t + 1.0))

    #         # MODIFIED: Add temporal weighting
    #         if t == 0:
    #             time_weight = 3.0  # Higher weight for t=0 (formation time)
    #         elif t <= 2:
    #             time_weight = 2.0  # Medium weight for early timesteps
    #         else:
    #             time_weight = 1.0  # Standard weight for later timesteps
            
    #         total_penalty = total_penalty + time_weight * (bond_pen_t + brid_pen_t)

    #         # logs
    #         tot_exp_hidden_bond += float(exp_hidden_bond_t.item())
    #         tot_exp_hidden_brid += float(exp_hidden_brid_t.item())
    #         tot_pred_bond += float(pred_bond_t.item())
    #         tot_pred_brid += float(pred_brid_t.item())
    #         tot_cand_pairs += n_cand_t

    #     final_penalty = penalty_strength * (total_penalty / float(max_timestep + 1))

    #     if neighbor_index is not None:
    #         print(f"[SparsePenalty] avg candidate pairs per t ≈ {tot_cand_pairs / float(max_timestep + 1):.1f}")
    #     print(f"  Expected hidden (candidates): bonding={tot_exp_hidden_bond:.1f}, bridging={tot_exp_hidden_brid:.1f}")
    #     print(f"  Predicted (candidates):       bonding={tot_pred_bond:.1f}, bridging={tot_pred_brid:.1f}")
    #     print(f"  Final penalty: {float(final_penalty.item()):.3f}")

    #     return final_penalty


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

    # def compute_sparsity_penalty(self, marginal_probs, network_data, node_batch, max_timestep, sparsity_strength=5.0):
    #     """
    #     Sparsity prior: penalize predicting links for HIDDEN pairs only.
    #     """
    #     total_prior = 0.0
    #     count = 0
    #     batch_nodes_set = set(node_batch.tolist())
        
    #     for pair_key, marginal in marginal_probs.items():
    #         parts = pair_key.split('_')
    #         if len(parts) != 3:
    #             continue
    #         i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
            
    #         if i not in batch_nodes_set and j not in batch_nodes_set:
    #             continue
    #         if t > max_timestep:
    #             continue
            
    #         # Skip observed pairs
    #         if network_data.is_observed(i, j, t):
    #             continue
            
    #         # Only penalize hidden pairs
    #         prior = (marginal[0] * torch.log(torch.tensor(0.99)) +
    #                 (marginal[1] + marginal[2]) * torch.log(torch.tensor(0.01)))
    #         total_prior += prior
    #         count += 1
        
    #     if count > 0:
    #         total_prior = total_prior / count
        
    #     return -sparsity_strength * total_prior

    # def compute_sparsity_penalty(self, marginal_probs, network_data, node_batch, 
    #                                  max_timestep, target_link_count=1800):
    #     """
    #     Adaptive: adjust penalty based on current predictions.
    #     """
    #     # Count current predicted links
    #     total_links = 0
    #     total_pairs = 0
        
    #     for pair_key, marginal in marginal_probs.items():
    #         parts = pair_key.split('_')
    #         if len(parts) != 3:
    #             continue
    #         i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
    #         if t > max_timestep:
    #             continue
    #         if network_data.is_observed(i, j, t):
    #             continue
            
    #         total_links += (marginal[1] + marginal[2]).item()
    #         total_pairs += 1
        
    #     # Adaptive strength based on current state
    #     if total_links < target_link_count * 0.5:
    #         # Too sparse, encourage links
    #         strength = 0.0  # No penalty
    #     elif total_links > target_link_count * 2:
    #         # Too dense, penalize links
    #         strength = 3.0
    #     else:
    #         # Reasonable range, moderate penalty
    #         strength = 1.0
        
    #     # Compute penalty
    #     total_prior = 0.0
    #     count = 0
    #     for pair_key, marginal in marginal_probs.items():
    #         parts = pair_key.split('_')
    #         if len(parts) != 3:
    #             continue
    #         i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
    #         if t > max_timestep or network_data.is_observed(i, j, t):
    #             continue
            
    #         prior = (marginal[0] * torch.log(torch.tensor(0.99)) +
    #                 (marginal[1] + marginal[2]) * torch.log(torch.tensor(0.01)))
    #         total_prior += prior
    #         count += 1
        
    #     if count > 0:
    #         total_prior = total_prior / count
        
    #     return -strength * total_prior


    
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
                    'observation': 20.0, 
                    'prior': 1.0,                       
                    'entropy': 1.0, 
                    'sparse_penalty': 0.0,    
                    'density_penalty': 1.0,    
                }
            elif epoch < 350:
                progress = (epoch - 150) / 200
                state_weight = 0.1  
                prior_weight = 1.0 + 9.0 * progress 
                observation_weight = 20.0 + 20.0 * progress
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
                    'observation': 40.0, 
                    'prior': 10.0,    
                    'entropy': 1.0, 
                    'sparse_penalty': 0.0,
                    'density_penalty': 10.0  # 🔧 FIX: Keep at 10 (half of obs)
                }
            else:  # epoch >= 550: Prediction learning phase
                # 🔧 STRONG SIGNAL: Network is frozen, give strong signals immediately
                # No need for gradual ramp-up since network structure won't be affected
                return {
                    'state': 200.0,                    # Strong signal for one-step prediction
                    'rollout': 200.0,                  # Strong signal for multi-step rollout
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
        if current_epoch >= 0:
            rollout_likelihood_raw, rollout_count = torch.tensor(0.0), 0
        else:
            # 🔧 IMPROVED: Remove hardcoded parameters, let function decide based on epoch
            rollout_likelihood_raw, rollout_count = self.compute_rollout_likelihood(
                features=features,
                initial_states=states[:, 0:1, :],  # [N, 1, 3] - only initial state
                distances=distances,
                network_data=network_data,
                gumbel_samples=gumbel_samples[0],      # Single sample dict
                max_timestep=max_timestep,
                true_states=states                 # [N, T+1, 3] - full ground truth
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

        print(f"Per-unit values - State: {state_likelihood_per_prediction.item():.4f}, "
              f"Obs: {network_likelihood_per_pair.item():.4f}, "
              f"Prior: {prior_likelihood_per_pair.item():.4f}, "
              f"Entropy: {entropy_per_pair.item():.4f}")

        weight = get_dynamic_weights(current_epoch)
        # Apply adaptive weighting
        weighted_state_likelihood = weight['state'] * state_likelihood_per_prediction
        weighted_rollout_likelihood = weight.get('rollout', 0.0) * rollout_likelihood_per_prediction
        weighted_observation_likelihood = weight['observation'] * network_likelihood_per_pair
        weighted_prior_likelihood = weight['prior'] * prior_likelihood_per_pair
        weighted_posterior_entropy = weight['entropy'] * entropy_per_pair
        weighted_density_penalty = weight['density_penalty'] * timestep_density_penalty
        weighted_sparsity_penalty = weight['sparse_penalty'] * sparsity_penalty

        
        # Total weighted ELBO
        # total_elbo = (weighted_state_likelihood + weighted_observation_likelihood +
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg - weighted_constraint_penalty)
        # total_elbo = (weighted_state_likelihood + weighted_observation_likelihood +
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg -
        #             weighted_constraint_penalty - weighted_density_penalty - weighted_info_propagation)
        # total_elbo = (weighted_state_likelihood + weighted_rollout_likelihood + weighted_observation_likelihood +
        #             weighted_prior_likelihood + weighted_posterior_entropy - weighted_confidence_reg 
        #              - weighted_density_penalty - weighted_info_propagation)
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
        params_influence = [p for p in self.state_transition.seq_pair_infl_nn.parameters() if p.requires_grad]
        params_self = [p for p in self.state_transition.seq_self_nn.parameters() if p.requires_grad]
        
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