import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import defaultdict

from .neural_networks import InfluenceNN, SelfActivationNN
from .utils import get_state_history_excluding_k, get_full_state_history

class StateTransition:
    """State transition with FR-SIC process and lingering influence support"""
    
    def __init__(self, self_nn: SelfActivationNN, influence_nn: InfluenceNN, L: int = 1,
                 L_linger: int = 3, decay_type: str = 'exponential', decay_rate: float = 0.5,
                 max_neighbor_influences: int = 10):
        self.self_nn = self_nn
        self.influence_nn = influence_nn
        self.L = L
        
        # NEW: Lingering influence parameters
        self.L_linger = L_linger          # How many timesteps influence persists
        self.decay_type = decay_type      # 'exponential', 'linear', or 'step'
        self.decay_rate = decay_rate      # Decay speed parameter
        
        # NEW: Link break history tracking
        # Structure: {household_i: [{'neighbor': j, 'link_type': 2, 'break_time': t}, ...]}
        self.broken_links_history = defaultdict(list)

        # NEW: Influence usage tracking
        # Structure: {(active_household, target_household, decision_type): usage_count}
        self.max_neighbor_influences = max_neighbor_influences
        self.influence_usage_history = defaultdict(int)
        # self.influence_stats = {
        # 'total_attempts': 0,
        # 'blocked_attempts': 0,
        # 'current_timestep': -1
        # }
    
    def _compute_inactive_fraction(self, states: torch.Tensor, time: int, decision_type: int) -> float:
        """
        Compute fraction of population still inactive for given decision type at time t.
        This is non-leaking: only uses current and past information.
        
        Args:
            states: [N, T, 3] - state tensor
            time: current timestep
            decision_type: 0 (vacant), 1 (repair), 2 (sell)
        
        Returns:
            fraction: float in [0, 1] - fraction of population inactive at time t
        """
        total_population = states.shape[0]
        current_inactive = torch.sum(states[:, time, decision_type] == 0).item()
        inactive_fraction = current_inactive / total_population
        return inactive_fraction
        
    def add_broken_link(self, household_i: int, household_j: int, link_type: int, break_time: int):
        """Record a broken link for both households"""
        # Add to both households' history (since links are bidirectional)

        # print(f"ADDING BROKEN LINK: {household_i}-{household_j}, break_time={break_time}")
        # import traceback
        # traceback.print_stack()

        self.broken_links_history[household_i].append({
            'neighbor': household_j,
            'link_type': link_type,
            'break_time': break_time
        })
        self.broken_links_history[household_j].append({
            'neighbor': household_i, 
            'link_type': link_type,
            'break_time': break_time
        })
    
    def cleanup_expired_history(self, current_time: int):
        """Remove link break records older than L_linger timesteps"""
        for household_i in list(self.broken_links_history.keys()):
            # Keep only recent breaks
            self.broken_links_history[household_i] = [
                record for record in self.broken_links_history[household_i]
                if current_time - record['break_time'] < self.L_linger
            ]
            # Remove empty entries
            if not self.broken_links_history[household_i]:
                del self.broken_links_history[household_i]
    
    def get_decay_weight(self, time_since_break: int) -> float:
        # """Calculate decay weight"""
        # if time_since_break >= self.L_linger:
        #     return 0.0
        # if self.decay_type == 'exponential':
        #     # Convert to tensor first, then back to float
        #     decay_tensor = torch.tensor(-self.decay_rate * time_since_break)
        #     return torch.exp(decay_tensor).item()
        # elif self.decay_type == 'linear':
        #     return max(0.0, 1.0 - time_since_break / self.L_linger)
        # else:  # step
        #     return 1.0
        # print(f"DEBUG: time_since_break={time_since_break}, decay_rate={self.decay_rate}")
    
        if time_since_break >= self.L_linger:
            return 0.0
        if self.decay_type == 'exponential':
            exponent = -self.decay_rate * time_since_break
            # print(f"DEBUG: exponent={exponent}")
            
            if exponent > 5.0:  
                print(f"WARNING: Exponent too large: {exponent}, clamping to 0.0")
                return 0.0
                
            decay_tensor = torch.tensor(exponent)
            result = torch.exp(decay_tensor).item()
            # print(f"DEBUG: decay_weight={result}")
            return result
    
    def detect_and_record_link_breaks(self, network_data, gumbel_samples: Dict[str, torch.Tensor], 
                                    current_time: int):
        """
        Detect link breaks from network evolution and record them in history.
        This should be called after network state updates but before computing influences.
        """
        if current_time == 0:
            return  # No breaks possible at t=0
        
        # Get current sample (assume we're using the first sample for break detection)
        current_sample = gumbel_samples[0] if isinstance(gumbel_samples, list) else gumbel_samples
        
        # Check all pairs for potential breaks
        for pair_key, link_probs in current_sample.items():
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            
            i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
            if t != current_time:
                continue
            
            # Get current link type (most probable)
            current_link_type = torch.argmax(link_probs).item()
            
            # Check if this was a bridging link at previous timestep
            prev_pair_key = f"{i}_{j}_{current_time-1}"
            if prev_pair_key in current_sample:
                prev_link_type = torch.argmax(current_sample[prev_pair_key]).item()
                
                # If was bridging (type 2) and now is no link (type 0), record break
                if prev_link_type == 2 and current_link_type == 0:
                    self.add_broken_link(i, j, prev_link_type, current_time)
                    # print(f"Recorded bridging link break: {i}-{j} at t={current_time}")
    
    def compute_lingering_influence_probability(self, household_idx: torch.Tensor, decision_type: int,
                                              features: torch.Tensor, states: torch.Tensor, 
                                              distances: torch.Tensor, time: int) -> torch.Tensor:
        """
        Compute lingering influence from recently broken links.
        
        Args:
            household_idx: Target households to compute influence for
            decision_type: Decision type (0=vacant, 1=repair, 2=sell)
            features, states, distances: Same as original function
            time: Current timestep
            
        Returns:
            lingering_influence_probs: [batch_size] tensor of lingering influence probabilities
        """
        # print(f"=== Lingering influence computation at time={time} ===")
        # total_records = sum(len(records) for records in self.broken_links_history.values())
        # print(f"Total history records: {total_records}")
        
        # for hh, records in self.broken_links_history.items():
        #     for record in records:
        #         break_time = record['break_time']
        #         time_since_break = time - break_time
        #         print(f"Record: hh={hh}, neighbor={record['neighbor']}, break_time={break_time}, time_since_break={time_since_break}")



        batch_size = len(household_idx)
        lingering_influence_probs = torch.ones(batch_size, dtype=torch.float32)  # Start with 1.0 (no influence)
        
        # Process each household in the batch
        for batch_idx, hh_i in enumerate(household_idx):
            hh_i_val = hh_i.item()
            
            # Get broken link history for this household
            if hh_i_val not in self.broken_links_history:
                continue  # No broken links, keep influence prob at 1.0
            
            # Process each broken link
            household_lingering_product = 1.0  # Product of (1 - lingering_influence)
            
            for broken_link in self.broken_links_history[hh_i_val]:
                neighbor_j = broken_link['neighbor']
                link_type = broken_link['link_type']
                break_time = broken_link['break_time']
                
                time_since_break = time - break_time
                if time_since_break >= self.L_linger:
                    continue  # Too old, no lingering influence
                
                # Check if the former neighbor is active for this decision type
                if states[neighbor_j, time, decision_type] != 1:
                    continue  # Former neighbor not active, no influence
                
                # Compute decay weight
                decay_weight = self.get_decay_weight(time_since_break)
                if decay_weight <= 0:
                    continue
                
                # Compute original influence strength (same as active neighbor influence)
                # Create dummy link representation for the broken link
                link_repr = torch.zeros(1, 3)
                link_repr[0, link_type] = 1.0  # One-hot encoding of original link type
                
                # Get state histories
                j_state_hist = get_full_state_history([neighbor_j], states, time, self.L)
                i_state_hist = get_state_history_excluding_k([hh_i_val], decision_type, states, time, self.L)
                
                # Features and distances
                feat_i = features[hh_i_val].unsqueeze(0)
                feat_j = features[neighbor_j].unsqueeze(0)
                dist = distances[hh_i_val, neighbor_j].unsqueeze(0).unsqueeze(1)
                
                # Decision type and time tensors
                decision_onehot = F.one_hot(torch.tensor(decision_type), num_classes=3).float().unsqueeze(0)
                time_tensor = torch.full((1, 1), time, dtype=torch.float32)
                
                # Inactive fraction for enhanced features
                inactive_frac = self._compute_inactive_fraction(states, time, decision_type)
                inactive_frac_tensor = torch.full((1, 1), inactive_frac, dtype=torch.float32)
                
                # Compute base influence probability with enhanced features
                base_influence_prob = self.influence_nn(
                    link_repr, j_state_hist, i_state_hist,
                    feat_i, feat_j, dist, decision_onehot, time_tensor,
                    inactive_fraction=inactive_frac_tensor
                ).squeeze().item()
                
                # Apply decay weight
                lingering_influence_prob = decay_weight * base_influence_prob
                
                # Update product term
                household_lingering_product *= (1 - lingering_influence_prob)
                
                # print(f"Lingering influence: {hh_i_val} <- {neighbor_j}, "
                #       f"decay_weight={decay_weight:.3f}, "
                #       f"base_influence={base_influence_prob:.3f}, "
                #       f"lingering_influence={lingering_influence_prob:.3f}")
            
            # Convert product to final influence probability
            lingering_influence_probs[batch_idx] = household_lingering_product
        
        return lingering_influence_probs
    
    def clear_influence_tracking(self):
        """Clear influence tracking at start of new simulation/evaluation"""
        self.influence_usage_history.clear()

    def can_influence(self, active_household: int, target_household: int, decision_type: int) -> bool:
        """Check if active household can still influence target for this decision type"""
        key = (active_household, target_household, decision_type)
        can_do = self.influence_usage_history[key] < self.max_neighbor_influences
    
        # if not can_do:
        #     self.influence_stats['blocked_attempts'] += 1
        
        return can_do

    def record_influence_attempt(self, active_household: int, target_household: int, decision_type: int):
        """Record that an influence attempt occurred"""
        key = (active_household, target_household, decision_type)
        self.influence_usage_history[key] += 1
        # self.influence_stats['total_attempts'] += 1

    def print_influence_stats(self, timestep: int):
        if timestep != self.influence_stats['current_timestep']:
            if self.influence_stats['total_attempts'] > 0:
                blocked_rate = self.influence_stats['blocked_attempts'] / self.influence_stats['total_attempts']
                print(f"t={timestep}: Total={self.influence_stats['total_attempts']}, "
                    f"Blocked={self.influence_stats['blocked_attempts']}, "
                    f"Rate={blocked_rate:.3f}")
            self.influence_stats['current_timestep'] = timestep
    
    def compute_activation_probability_with_lingering(self, household_idx: torch.Tensor, decision_type: int,
                                                features: torch.Tensor, states: torch.Tensor, 
                                                distances: torch.Tensor, network_data,
                                                gumbel_samples: Dict[str, torch.Tensor], time: int,
                                                return_components: bool = False):
        """
        Enhanced FR-SIC process with lingering influence from broken links.
        
        Modified FR-SIC formula:
        P(activate) = 1 - (1 - p_self) × (1 - p_current_neighbors) × (1 - p_lingering_influence)
        
        Args:
            return_components: If True, return dict with probability components for diagnostic
        """
        batch_size = len(household_idx)
        
        if batch_size == 0:
            if return_components:
                return torch.tensor([]), {}
            return torch.tensor([])
        
        # Step 1: Compute self-activation with enhanced features
        hh_features = features[household_idx]
        state_history_excluding_k = self._batch_get_state_history_excluding_k(
            household_idx, decision_type, states, time, self.L
        )
        
        decision_onehot = F.one_hot(torch.tensor(decision_type), num_classes=3).float()
        decision_onehot_batch = decision_onehot.unsqueeze(0).expand(batch_size, -1)
        time_tensor_batch = torch.full((batch_size, 1), time, dtype=torch.float32)
        
        # Compute inactive fraction (population saturation info)
        inactive_frac = self._compute_inactive_fraction(states, time, decision_type)
        inactive_frac_batch = torch.full((batch_size, 1), inactive_frac, dtype=torch.float32)
        
        # Call self_nn with enhanced features (will use them if use_enhanced_features=True)
        p_self = self.self_nn(hh_features, state_history_excluding_k, 
                             decision_onehot_batch, time_tensor_batch, 
                             inactive_fraction=inactive_frac_batch).squeeze(1)
        
        # Step 2: Compute current neighbor influence (existing logic)
        neighbor_data = self._precompute_neighbor_relationships(
            household_idx, states, time, decision_type, features.shape[0], network_data
        )
        
        current_neighbor_influence_probs = torch.ones(batch_size, dtype=torch.float32)
        if len(neighbor_data['all_pairs']) > 0:
            influence_probs = self._compute_batch_neighbor_influence(
                neighbor_data, features, states, distances, network_data,
                gumbel_samples, time, decision_type
            )
            current_neighbor_influence_probs = self._aggregate_neighbor_influences(
                influence_probs, neighbor_data, batch_size
            )
        
        # Step 3: Compute lingering influence (NEW)
        lingering_influence_probs = self.compute_lingering_influence_probability(
            household_idx, decision_type, features, states, distances, time
        )
        
        # Step 4: Combined FR-SIC formula with lingering influence
        # P(activate) = 1 - (1 - p_self) × current_neighbor_product × lingering_product
        activation_probs = 1 - (1 - p_self) * current_neighbor_influence_probs * lingering_influence_probs
        
        if return_components:
            components = {
                'p_self': p_self,
                'current_neighbor_product': current_neighbor_influence_probs,
                'lingering_product': lingering_influence_probs,
                'final_probs': activation_probs,
                'num_active_neighbors': torch.tensor([len(neighbor_data['all_pairs'])], dtype=torch.float32)
            }
            return activation_probs, components
        
        return activation_probs
    
    # Update the main compute_activation_probability method to use the new enhanced version
    def compute_activation_probability(self, household_idx: torch.Tensor, decision_type: int,
                                     features: torch.Tensor, states: torch.Tensor, distances: torch.Tensor,
                                     network_data, gumbel_samples: Dict[str, torch.Tensor], time: int,
                                     return_components: bool = False):
        """
        Main activation probability computation - now includes lingering influence.
        
        Args:
            return_components: If True, return (probs, components_dict) for diagnostic
        """
        # First, detect and record any new link breaks
        self.detect_and_record_link_breaks(network_data, gumbel_samples, time)
        
        # Clean up old history entries
        self.cleanup_expired_history(time)

        # if time % 2 == 0:
        #     self.print_influence_stats(time)
        
        # Compute activation probabilities with lingering influence
        return self.compute_activation_probability_with_lingering(
            household_idx, decision_type, features, states, distances,
            network_data, gumbel_samples, time, return_components=return_components
        )
    
    def _batch_get_state_history_excluding_k(self, household_idx: torch.Tensor, decision_type: int,
                                            states: torch.Tensor, time: int, L: int):
        """
        OPTIMIZED: Batch computation of state histories to avoid repeated calls.
        """
        batch_size = len(household_idx)
        start_time = max(0, time - L + 1)
        end_time = min(time + 1, states.shape[1])
        
        # Batch extract state histories
        # OPTIMIZATION: Detach historical states (they are observed data, no gradient needed)
        # This significantly reduces computational graph complexity for backward pass
        state_hist = states[household_idx, start_time:end_time, :].detach()  # [batch_size, time_steps, 3]
        
        # Remove decision_type dimension
        other_decisions = [i for i in range(3) if i != decision_type]
        state_hist_excluding_k = state_hist[:, :, other_decisions]  # [batch_size, time_steps, 2]
        
        # Calculate expected length and pad if necessary
        expected_length = L * 2  # L timesteps × 2 decision types
        actual_timesteps = state_hist_excluding_k.shape[1]
        actual_length = actual_timesteps * 2
        
        # Flatten
        state_hist_flat = state_hist_excluding_k.view(batch_size, -1)
        
        # Pad if necessary
        if actual_length < expected_length:
            padding_length = expected_length - actual_length
            padding = torch.zeros(batch_size, padding_length, 
                                dtype=state_hist.dtype, device=state_hist.device)
            state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
        
        return state_hist_flat
    
    def _precompute_neighbor_relationships(self, household_idx, states, time, decision_type,
                                       n_households, network_data):
        """
        Build neighbor lists for each i in the batch, but ONLY over candidate neighbors.
        """
        active_state_mask = states[:, time, decision_type] == 1  # [N]
        neighbor_index = getattr(network_data, "neighbor_index", None)

        all_pairs = []
        household_to_pairs_map = {}
        pair_to_households = []

        for batch_idx, hh_i in enumerate(household_idx):
            i = hh_i.item()
            household_to_pairs_map[batch_idx] = []

            # candidate neighbors for i
            candidates = range(n_households) if neighbor_index is None else neighbor_index[i]

            for j in candidates:
                if j == i:
                    continue
                if active_state_mask[j]:
                    if self.can_influence(j, i, decision_type):
                        pid = len(all_pairs)
                        all_pairs.append((i, j))
                        household_to_pairs_map[batch_idx].append(pid)
                        pair_to_households.append((batch_idx, j))

        return {
            'all_pairs': all_pairs,
            'household_to_pairs_map': household_to_pairs_map,
            'pair_to_households': pair_to_households
        }
    
    def _compute_batch_neighbor_influence(self, neighbor_data: dict, features: torch.Tensor,
                                        states: torch.Tensor, distances: torch.Tensor,
                                        network_data, gumbel_samples: Dict[str, torch.Tensor],
                                        time: int, decision_type: int):
        """
        OPTIMIZED: Compute all neighbor influences in a single batch call.
        """
        all_pairs = neighbor_data['all_pairs']
        pair_to_households = neighbor_data['pair_to_households']
        
        if len(all_pairs) == 0:
            return torch.tensor([])
        
        n_pairs = len(all_pairs)
        
        # Pre-allocate tensors for batch processing
        batch_link_reprs = []
        batch_j_state_hists = []
        batch_i_state_hists = []
        batch_feat_i = []
        batch_feat_j = []
        batch_distances = []
        
        # Batch collect all neighbor state histories
        j_indices = [j for _, j in pair_to_households]
        if len(j_indices) > 0:
            j_indices_tensor = torch.tensor(j_indices, dtype=torch.long)
            batch_j_state_hists_tensor = self._batch_get_full_state_history(
                j_indices_tensor, states, time, self.L
            )
        
        # Process each pair efficiently
        for pair_idx, ((hh_i, j), (batch_idx, _)) in enumerate(zip(all_pairs, pair_to_households)):
            # Link representation
            link_repr = self._get_link_representation(hh_i, j, network_data, gumbel_samples, time)
            batch_link_reprs.append(link_repr.squeeze(0))
            
            # State histories (j already computed in batch)
            batch_j_state_hists.append(batch_j_state_hists_tensor[pair_idx])
            
            # i state history (compute individually for now, could be optimized further)
            i_state_hist = get_state_history_excluding_k([hh_i], decision_type, states, time, self.L)
            batch_i_state_hists.append(i_state_hist.squeeze(0))
            
            # Features and distances
            batch_feat_i.append(features[hh_i])
            batch_feat_j.append(features[j])
            batch_distances.append(distances[hh_i, j].unsqueeze(0))
        
        if len(batch_link_reprs) == 0:
            return torch.tensor([])
        
        # Convert to tensors
        link_reprs_tensor = torch.stack(batch_link_reprs)  # [n_pairs, 3]
        j_state_hists_tensor = torch.stack(batch_j_state_hists)  # [n_pairs, L*3]
        i_state_hists_tensor = torch.stack(batch_i_state_hists)  # [n_pairs, L*2]
        feat_i_tensor = torch.stack(batch_feat_i)  # [n_pairs, feature_dim]
        feat_j_tensor = torch.stack(batch_feat_j)  # [n_pairs, feature_dim]
        distances_tensor = torch.stack(batch_distances)  # [n_pairs, 1]
        
        # Common tensors for the batch
        decision_onehot_tensor = F.one_hot(torch.tensor(decision_type), num_classes=3).float()
        decision_onehot_batch = decision_onehot_tensor.unsqueeze(0).expand(n_pairs, -1)
        time_tensor_batch = torch.full((n_pairs, 1), time, dtype=torch.float32)
        
        # Compute inactive fraction for enhanced features
        inactive_frac = self._compute_inactive_fraction(states, time, decision_type)
        inactive_frac_batch = torch.full((n_pairs, 1), inactive_frac, dtype=torch.float32)
        
        # Single batch call to influence neural network with enhanced features
        influence_probs = self.influence_nn(
            link_reprs_tensor, j_state_hists_tensor, i_state_hists_tensor,
            feat_i_tensor, feat_j_tensor, distances_tensor,
            decision_onehot_batch, time_tensor_batch,
            inactive_fraction=inactive_frac_batch  # Enhanced feature
        ).squeeze(1)

        # ❌ DISABLED: Normalize by sqrt(n) - was preventing neighbor influence from accumulating
        # This normalization was too aggressive and made neighbor influence negligible
        # n_neighbors = len(influence_probs)
        # if n_neighbors > 1:
        #     scale_factor = 1.0 / torch.sqrt(torch.tensor(float(n_neighbors)))
        #     influence_probs = influence_probs * scale_factor

        # NEW: Record influence attempts for computed influences
        all_pairs = neighbor_data['all_pairs']
        pair_to_households = neighbor_data['pair_to_households']
        
        for pair_idx, ((hh_i, j), (batch_idx, _)) in enumerate(zip(all_pairs, pair_to_households)):
            # Record that j (active) attempted to influence hh_i (target)
            self.record_influence_attempt(j, hh_i, decision_type)
        
        return influence_probs
    
    def _batch_get_full_state_history(self, household_idx: torch.Tensor, states: torch.Tensor, 
                                     time: int, L: int):
        """
        OPTIMIZED: Batch version of get_full_state_history.
        """
        batch_size = len(household_idx)
        start_time = max(0, time - L + 1)
        end_time = min(time + 1, states.shape[1])
        
        # Batch extract state histories
        # OPTIMIZATION: Detach historical states (observed data, no gradient needed)
        state_hist = states[household_idx, start_time:end_time, :].detach()  # [batch_size, time_steps, 3]
        
        # Calculate expected length
        expected_length = L * 3  # L timesteps × 3 decision types
        actual_timesteps = state_hist.shape[1]
        actual_length = actual_timesteps * 3
        
        # Flatten
        state_hist_flat = state_hist.view(batch_size, -1)
        
        # Pad if necessary
        if actual_length < expected_length:
            padding_length = expected_length - actual_length
            padding = torch.zeros(batch_size, padding_length, 
                                dtype=state_hist.dtype, device=state_hist.device)
            state_hist_flat = torch.cat([padding, state_hist_flat], dim=1)
        
        return state_hist_flat
    

    def _get_link_representation(self, i, j, network_data, gumbel_samples, time):
        """Get continuous link representation."""
        i, j = min(i, j), max(i, j)
        
        if network_data.is_observed(i, j, time) & time<=15:
            link_type = network_data.get_link_type(i, j, time)
            link_repr = F.one_hot(torch.tensor(link_type), num_classes=3).float()
        else:
            pair_key = f"{i}_{j}_{time}"
            link_repr = gumbel_samples.get(pair_key, torch.tensor([1.0, 0.0, 0.0]))
        
        return link_repr.unsqueeze(0)
    
    def _aggregate_neighbor_influences(self, influence_probs: torch.Tensor, neighbor_data: dict, 
                                     batch_size: int):
        """
        OPTIMIZED: Aggregate neighbor influences using vectorized operations where possible.
        """
        household_to_pairs_map = neighbor_data['household_to_pairs_map']
        
        final_influence_probs = torch.ones(batch_size, dtype=torch.float32)
        
        for batch_idx in range(batch_size):
            pair_indices = household_to_pairs_map[batch_idx]
            
            if len(pair_indices) == 0:
                # No neighbors, influence probability remains 1.0
                continue
            
            # Get influence probabilities for this household's neighbors
            neighbor_probs = influence_probs[pair_indices]

            n_neighbors = len(neighbor_probs)
            # if n_neighbors > 1:
            #     decay_factor = 1.0 / torch.sqrt(torch.tensor(float(n_neighbors)))
            #     neighbor_probs = neighbor_probs * decay_factor 

            
            # Compute product term: ∏(1 - p_inf)
            log_term = torch.sum(torch.log1p(-neighbor_probs.clamp(0, 1-1e-8)))
            product_term = torch.exp(log_term)
            final_influence_probs[batch_idx] = product_term
        
        return final_influence_probs
    
    def compute_detailed_activation_probability(self, household_idx: torch.Tensor, decision_type: int,
                                    features: torch.Tensor, states: torch.Tensor, distances: torch.Tensor,
                                    network_data, gumbel_samples: Dict[str, torch.Tensor], time: int):
        """
        Detailed version that returns self-activation, neighbor-by-neighbor influence probabilities,
        and lingering influence from recently broken links.
        Only used for evaluation, not training.
        """
        batch_size = len(household_idx)
        n_households = features.shape[0]
        
        # First, detect and record any new link breaks
        self.detect_and_record_link_breaks(network_data, gumbel_samples, time)
        
        # Clean up old history entries
        self.cleanup_expired_history(time)
        
        detailed_results = []
        
        for batch_idx, hh_i in enumerate(household_idx):
            hh_i_val = hh_i.item()
            
            # 1. SELF-ACTIVATION PROBABILITY
            hh_features = features[hh_i_val].unsqueeze(0)
            state_history_excluding_k = get_state_history_excluding_k(
                [hh_i_val], decision_type, states, time, self.L
            )
            
            decision_onehot = F.one_hot(torch.tensor(decision_type), num_classes=3).float().unsqueeze(0)
            time_tensor = torch.full((1, 1), time, dtype=torch.float32)
            
            # Compute inactive fraction for enhanced features
            inactive_frac = self._compute_inactive_fraction(states, time, decision_type)
            inactive_frac_tensor = torch.full((1, 1), inactive_frac, dtype=torch.float32)
            
            p_self = self.self_nn(hh_features, state_history_excluding_k, decision_onehot, time_tensor,
                                 inactive_fraction=inactive_frac_tensor).squeeze().item()
            
            # 2. CURRENT NEIGHBOR-BY-NEIGHBOR INFLUENCE PROBABILITIES
            active_neighbors = []
            neighbor_influences = []

            for j in range(n_households):
                if j == hh_i_val:
                    continue
                    
                # First check if there's a connection
                link_repr = self._get_link_representation(hh_i_val, j, network_data, gumbel_samples, time)
                link_type = torch.argmax(link_repr).item()
                
                if link_type > 0:  # Has connection
                    # Then check if neighbor is active
                    if states[j, time, decision_type] == 1:  # Neighbor is active
                        if self.can_influence(j, hh_i_val, decision_type):
                            # Calculate influence probability
                            active_neighbors.append(j)
                            j_state_hist = get_full_state_history([j], states, time, self.L)
                            i_state_hist = get_state_history_excluding_k([hh_i_val], decision_type, states, time, self.L)
                            
                            feat_i = features[hh_i_val].unsqueeze(0)
                            feat_j = features[j].unsqueeze(0)
                            dist = distances[hh_i_val, j].unsqueeze(0).unsqueeze(1)
                            
                            p_inf = self.influence_nn(link_repr, j_state_hist, i_state_hist, 
                                                    feat_i, feat_j, dist, decision_onehot, time_tensor,
                                                    inactive_fraction=inactive_frac_tensor).squeeze().item()
                            
                            
                            self.record_influence_attempt(j, hh_i_val, decision_type)
                            neighbor_influences.append({
                                'neighbor_id': j,
                                'link_type': link_type,
                                'link_probs': link_repr.squeeze().tolist(),
                                'influence_prob': p_inf,
                                'distance': distances[hh_i_val, j].item(),
                                'influence_type': 'current_neighbor'
                            })

            # 3. LINGERING INFLUENCE FROM RECENTLY BROKEN LINKS
            lingering_influences = []
            
            if hh_i_val in self.broken_links_history:
                for broken_link in self.broken_links_history[hh_i_val]:
                    neighbor_j = broken_link['neighbor']
                    link_type = broken_link['link_type']
                    break_time = broken_link['break_time']
                    
                    time_since_break = time - break_time
                    if time_since_break >= self.L_linger:
                        continue  # Too old, no lingering influence
                    
                    # Check if the former neighbor is active for this decision type
                    if states[neighbor_j, time, decision_type] != 1:
                        continue  # Former neighbor not active, no influence
                    
                    # Compute decay weight
                    decay_weight = self.get_decay_weight(time_since_break)
                    if decay_weight <= 0:
                        continue
                    
                    # Compute original influence strength (same as active neighbor influence)
                    # Create dummy link representation for the broken link
                    link_repr = torch.zeros(1, 3)
                    link_repr[0, link_type] = 1.0  # One-hot encoding of original link type
                    
                    # Get state histories
                    j_state_hist = get_full_state_history([neighbor_j], states, time, self.L)
                    i_state_hist = get_state_history_excluding_k([hh_i_val], decision_type, states, time, self.L)
                    
                    # Features and distances
                    feat_i = features[hh_i_val].unsqueeze(0)
                    feat_j = features[neighbor_j].unsqueeze(0)
                    dist = distances[hh_i_val, neighbor_j].unsqueeze(0).unsqueeze(1)
                    
                    # Compute base influence probability with enhanced features
                    base_influence_prob = self.influence_nn(
                        link_repr, j_state_hist, i_state_hist,
                        feat_i, feat_j, dist, decision_onehot, time_tensor,
                        inactive_fraction=inactive_frac_tensor
                    ).squeeze().item()
                    
                    # Apply decay weight
                    lingering_influence_prob = decay_weight * base_influence_prob
                    
                    lingering_influences.append({
                        'neighbor_id': neighbor_j,
                        'original_link_type': link_type,
                        'break_time': break_time,
                        'time_since_break': time_since_break,
                        'decay_weight': decay_weight,
                        'base_influence_prob': base_influence_prob,
                        'influence_prob': lingering_influence_prob,
                        'distance': distances[hh_i_val, neighbor_j].item(),
                        'influence_type': 'lingering'
                    })
            
            # 4. COMPUTE FINAL PROBABILITY STEP BY STEP
            # Current neighbor social influence term
            if len(neighbor_influences) == 0:
                current_neighbor_product = 1.0
                current_social_influence_term = 0.0
            else:
                neighbor_probs = [ni['influence_prob'] for ni in neighbor_influences]
                current_neighbor_product = torch.prod(torch.tensor([1 - p for p in neighbor_probs])).item()
                current_social_influence_term = 1 - current_neighbor_product
            
            # Lingering influence term
            if len(lingering_influences) == 0:
                lingering_product = 1.0
                lingering_influence_term = 0.0
            else:
                lingering_probs = [li['influence_prob'] for li in lingering_influences]
                lingering_product = torch.prod(torch.tensor([1 - p for p in lingering_probs])).item()
                lingering_influence_term = 1 - lingering_product
            
            # Combined social influence (both current and lingering)
            total_social_product = current_neighbor_product * lingering_product
            total_social_influence_term = 1 - total_social_product
            
            # Final activation probability using enhanced FR-SIC formula
            final_activation_prob = 1 - (1 - p_self) * total_social_product
            
            # Combine all influences for detailed output
            all_influences = neighbor_influences + lingering_influences
            
            detailed_results.append({
                'household_id': hh_i_val,
                'decision_type': decision_type,
                'timestep': time + 1,
                'self_activation_prob': p_self,
                'active_neighbors': len(active_neighbors),
                'neighbor_influences': neighbor_influences,
                'lingering_influences': lingering_influences,
                'all_influences': all_influences,
                'current_social_influence_term': current_social_influence_term,
                'lingering_influence_term': lingering_influence_term,
                'total_social_influence_term': total_social_influence_term,
                'final_activation_prob': final_activation_prob,
                'computation_breakdown': {
                    'p_self': p_self,
                    'current_neighbor_product': current_neighbor_product,
                    'lingering_product': lingering_product,
                    'total_social_product': total_social_product,
                    'formula': f"1 - (1 - {p_self:.4f}) × {current_neighbor_product:.4f} × {lingering_product:.4f} = {final_activation_prob:.4f}"
                }
            })
        
        # Return both detailed results and the original activation probabilities
        final_probs = torch.tensor([r['final_activation_prob'] for r in detailed_results])
        return final_probs, detailed_results