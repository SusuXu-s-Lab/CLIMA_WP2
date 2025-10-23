import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from .elbo_computation import ELBOComputation
from .gumbel_softmax import GumbelSoftmaxSampler
from .variational_posterior import MeanFieldPosterior
from models.utils import build_neighbor_index_from_distances


class NetworkStateTrainer:
    """
    Updated training coordinator following PDF formulation.
    
    Key changes:
    1. Compute both conditional and marginal probabilities
    2. Use marginals for Gumbel-Softmax sampling
    3. Pass both to ELBO computation
    """
    
    def __init__(self, 
                 mean_field_posterior: MeanFieldPosterior,
                 gumbel_sampler: GumbelSoftmaxSampler,
                 elbo_computer: ELBOComputation,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 L_linger: int = 3,
                 decay_type: str = 'exponential',
                 decay_rate: float = 0.5):
        
        self.mean_field_posterior = mean_field_posterior
        self.gumbel_sampler = gumbel_sampler
        self.elbo_computer = elbo_computer
        self.L_linger = L_linger
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        
        # Collect all parameters
        all_params = []
        all_params.extend(self.mean_field_posterior.network_type_nn.parameters())
        all_params.extend(self.elbo_computer.state_transition.self_nn.parameters())
        all_params.extend(self.elbo_computer.state_transition.influence_nn.parameters())
        all_params.extend(self.elbo_computer.network_evolution.interaction_nn.parameters())
        all_params.extend(self.elbo_computer.network_evolution.parameters())
        # all_params.extend([self.elbo_computer.rho_1, self.elbo_computer.rho_2])
        
        self.optimizer = optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
        
        self.epoch = 0
        self.training_history = []
        
    # def temperature_schedule(self, epoch: int, max_epochs: int) -> float:
    #     progress = epoch / max_epochs
    #     if progress < 0.4:  
    #         return 2.0
    #     else:
    #         return 2.0 * (0.3 / 2.0) ** ((progress - 0.4) / 0.6)
        
    def temperature_schedule(self, epoch: int, max_epochs: int) -> float:

        if epoch < 50:
            return 2.0
        elif epoch < 200:
            return 1.5
        else:
            return 1.0
    
    def sample_schedule(self, epoch: int, max_epochs: int) -> int:
        if epoch < 50:
            return 2  
        elif epoch < max_epochs * 0.6:
            return 3 
        else:
            return 2  
        

    def monitor_network_distributions(self, marginal_probs):
        """Monitor network type distribution statistics"""
        if len(marginal_probs) == 0:
            print("  Network Distribution: No hidden pairs")
            return
        
        # Collect all marginal probabilities
        all_probs = []
        for pair_key, prob in marginal_probs.items():
            if isinstance(prob, torch.Tensor) and prob.dim() == 1 and prob.shape[0] == 3:
                all_probs.append(prob.detach().cpu())
        
        if len(all_probs) == 0:
            print("  Network Distribution: No valid probabilities")
            return
        
        # Convert to tensor and compute statistics
        all_probs = torch.stack(all_probs)  # [num_pairs, 3]
        
        # Calculate mean and std for each type
        no_link_stats = all_probs[:, 0]
        bonding_stats = all_probs[:, 1] 
        bridging_stats = all_probs[:, 2]
        
        print(f"  Network Type Distribution ({len(all_probs)} pairs):")
        print(f"    No Link:  μ={no_link_stats.mean():.3f}, σ={no_link_stats.std():.3f}, range=[{no_link_stats.min():.3f}, {no_link_stats.max():.3f}]")
        print(f"    Bonding:  μ={bonding_stats.mean():.3f}, σ={bonding_stats.std():.3f}, range=[{bonding_stats.min():.3f}, {bonding_stats.max():.3f}]")
        print(f"    Bridging: μ={bridging_stats.mean():.3f}, σ={bridging_stats.std():.3f}, range=[{bridging_stats.min():.3f}, {bridging_stats.max():.3f}]")
        
        # Additional statistics: most certain and uncertain pairs
        entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-8), dim=1)
        print(f"    Uncertainty: μ_entropy={entropy.mean():.3f}, σ_entropy={entropy.std():.3f}")
        
        # Statistics of dominant type distribution
        dominant_types = torch.argmax(all_probs, dim=1)
        type_counts = torch.bincount(dominant_types, minlength=3)
        type_pcts = type_counts.float() / len(all_probs) * 100
        print(f"    Dominant Types: No Link {type_pcts[0]:.1f}%, Bonding {type_pcts[1]:.1f}%, Bridging {type_pcts[2]:.1f}%")


    def monitor_link_prediction_performance(self, marginal_probs, ground_truth_network, observed_network, max_timestep):
        """
        Simple link prediction monitoring with detailed breakdown.

        Args:
            marginal_probs: Dict[str, torch.Tensor] - marginal probabilities from variational posterior
            ground_truth_network: NetworkData object with ground truth
            observed_network: NetworkData object with partial observations
            max_timestep: int - maximum timestep to evaluate
        
        Returns:
            Dict with precision, recall and additional diagnostics
        """
        if len(marginal_probs) == 0:
            return {'precision': 0.0, 'recall': 0.0}
        
        predicted_links = 0
        true_links = 0
        correct_links = 0
        correct_observed = 0
        correct_ever_observed = 0       
        correct_never_observed = 0     

        processed_pairs = set()

        for pair_key, marginal_prob in marginal_probs.items():
            # Parse pair_key: "i_j_t"
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            i, j, t = int(parts[0]), int(parts[1]), int(parts[2])

            pair_id = (min(i, j), max(i, j), t)
            if pair_id in processed_pairs or t > max_timestep:
                continue
            processed_pairs.add(pair_id)

            # Prediction: 1 if any connection (argmax > 0), 0 if no connection
            predicted_exists = 1 if torch.argmax(marginal_prob).item() > 0 else 0

            # Ground truth
            true_link_type = ground_truth_network.get_link_type(i, j, t)
            true_exists = 1 if true_link_type > 0 else 0

            if_observed = observed_network.is_observed(i, j, t)

            # Update counters
            if predicted_exists == 1:
                predicted_links += 1
            if true_exists == 1:
                true_links += 1
            if predicted_exists == 1 and true_exists == 1:
                correct_links += 1
                if if_observed == 1:
                    correct_observed += 1

                ever_observed = any(
                    observed_network.is_observed(i, j, τ)
                    for τ in range(max_timestep + 1)
                )
                if ever_observed:
                    correct_ever_observed += 1
                else:
                    correct_never_observed += 1

        # Calculate metrics
        precision = correct_links / predicted_links if predicted_links > 0 else 0.0
        recall = correct_links / true_links if true_links > 0 else 0.0

        print(f"\n[Link Prediction Details]")
        print(f"  Ground truth edges: {true_links}")
        print(f"  Predicted edges: {predicted_links}")
        print(f"  Correct predictions: {correct_links}")
        print(f"  Correct observed predictions (current t): {correct_observed}")
        print(f"  Correct predictions ever observed before: {correct_ever_observed}")
        print(f"  Correct predictions never observed before: {correct_never_observed}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")

        return {
            'precision': precision,
            'recall': recall,
            'correct_observed': correct_observed,
            'correct_ever_observed': correct_ever_observed,
            'correct_never_observed': correct_never_observed
        }


    
    def train_epoch_batched(self,
                           features: torch.Tensor,
                           states: torch.Tensor,
                           distances: torch.Tensor,
                           network_data,
                           ground_truth_network,
                           node_batches: List[torch.Tensor],
                           max_timestep: int,
                           max_epochs: int,
                           lambda_constraint: float = 0.01) -> Dict[str, float]:
        """
        Train one epoch with updated formulation.
        
        Key changes:
        1. Compute both conditional and marginal probabilities
        2. Use marginals for sampling
        3. Pass both to ELBO computation
        """
        if self.epoch >= 550:
            for param in self.mean_field_posterior.network_type_nn.parameters():
                param.requires_grad = False
            for param in self.elbo_computer.network_evolution.parameters():
                param.requires_grad = False
        
        
        temperature = self.temperature_schedule(self.epoch, max_epochs)

        base_tau = self.temperature_schedule(self.epoch, max_epochs)
        if len(self.training_history) > 0:
            prev_entropy = float(self.training_history[-1].get('posterior_entropy', 0.08))
        else:
            prev_entropy = 0.08
     
        if prev_entropy < 0.05:
            temperature = min(base_tau * 1.4, 2.0)   # 稍加热
        elif prev_entropy < 0.10:
            temperature = min(base_tau * 1.2, 1.8)
        elif prev_entropy > 0.20:
            temperature = max(base_tau * 0.9, 0.5)   # 稍降温
       
        if prev_entropy < 0.05:
            self.gumbel_sampler.noise_scale = min(getattr(self.gumbel_sampler, 'noise_scale', 1.0) * 1.2, 2.0)
        elif prev_entropy > 0.20:
            self.gumbel_sampler.noise_scale = max(getattr(self.gumbel_sampler, 'noise_scale', 1.0) * 0.9, 0.8)
        else:
            self.gumbel_sampler.noise_scale = getattr(self.gumbel_sampler, 'noise_scale', 1.0)

        num_samples = self.sample_schedule(self.epoch, max_epochs)

        
        # Accumulate metrics across batches
        total_metrics = {
            'total_elbo': 0.0,
            'state_likelihood': 0.0,
            'observation_likelihood': 0.0,
            'rollout_likelihood': 0.0,
            'prior_likelihood': 0.0,
            'posterior_entropy': 0.0,
            # 'confidence_regularization': 0.0,
            # 'constraint_penalty': 0.0
        }
        
        self.elbo_computer.state_transition.clear_influence_tracking()
        
        # Process each batch
        for batch_idx, node_batch in enumerate(node_batches):
            
            # Compute BOTH conditional and marginal probabilities
            conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
                    features, states, distances, node_batches[0], network_data, max_timestep)
            print(f"Conditional and marginal probabilities finished for batch {batch_idx + 1}/{len(node_batches)}")

            # Monitor link prediction performance 
            if batch_idx == 0:  # Only check first batch
                link_metrics = self.monitor_link_prediction_performance(
                    marginal_probs, ground_truth_network, network_data, max_timestep
                )

            if self.epoch < 10 and self.epoch % 1 == 0:
                print(f"\n=== Epoch {self.epoch} Network Distribution Summary ===")
                # Recompute to get statistics (only use first batch)
                self.monitor_network_distributions(marginal_probs)
            elif self.epoch < 50 and  self.epoch % 1 == 0:
                print(f"\n=== Epoch {self.epoch} Network Distribution Summary ===")
                # Recompute to get statistics (only use first batch)
                self.monitor_network_distributions(marginal_probs)
            elif self.epoch % 2 == 0:
                print(f"\n=== Epoch {self.epoch} Network Distribution Summary ===")
                # Recompute to get statistics (only use first batch)
                self.monitor_network_distributions(marginal_probs)
                       
            # Sample hidden links using MARGINAL probabilities
            gumbel_samples = self.gumbel_sampler.sample_hidden_links_batch(
                marginal_probs, temperature, num_samples
            )
            print(f"Gumbel sampling finished for batch {batch_idx + 1}/{len(node_batches)}")

            self.elbo_computer.state_transition.broken_links_history.clear()
            
            # Compute ELBO using BOTH conditional and marginal probabilities
            batch_elbo = self.elbo_computer.compute_elbo_batch(
                            features, states, distances, node_batch, network_data,
                            conditional_probs, marginal_probs, gumbel_samples, max_timestep, 
                            lambda_constraint, current_epoch=self.epoch
                        )
            print(f"ELBO computation finished for batch {batch_idx + 1}/{len(node_batches)}")
            
            # Backward pass (accumulate gradients)
            batch_loss = -batch_elbo['total_elbo']  # Maximize ELBO = minimize negative ELBO
            # batch_loss.backward()
            # self.optimizer.step()  # Update parameters
            # self.optimizer.zero_grad()  # Reset gradients for next batch
            print(f"Backward pass finished for batch {batch_idx + 1}/{len(node_batches)}")
        
            
            # Accumulate metrics
            for key in total_metrics.keys():
                total_metrics[key] += batch_elbo[key].item()
        
        del batch_elbo  # 只删除局部变量
        del conditional_probs
        del marginal_probs  
        del gumbel_samples
        
        if batch_idx % 2 == 0:
            import gc
            gc.collect()
        
        # Update parameters after all batches
        self.optimizer.zero_grad()
        batch_loss.backward()

        # === Gradient diagnostics for PREDICTION networks only ===
        with torch.no_grad():
            try:
                print("\n=== PREDICTION NN DIAGNOSTICS (InfluenceNN & SelfNN) ===")
                
                # Helper function to compute gradient and parameter statistics
                def compute_stats(module, name):
                    params = list(module.parameters())
                    if len(params) == 0:
                        print(f"[{name}] No parameters")
                        return
                    
                    # Gradient statistics
                    grad_norms = []
                    for p in params:
                        if p.grad is not None:
                            grad_norms.append(p.grad.norm().item())
                    
                    if len(grad_norms) > 0:
                        total_grad_norm = sum(g**2 for g in grad_norms) ** 0.5
                        max_grad = max(grad_norms)
                        min_grad = min(grad_norms)
                        print(f"[{name}] Grad - Total: {total_grad_norm:.6f}, Max: {max_grad:.6f}, Min: {min_grad:.6f}")
                    else:
                        print(f"[{name}] Grad - NONE or all zeros")
                    
                    # Parameter statistics
                    param_values = torch.cat([p.flatten() for p in params])
                    param_mean = param_values.mean().item()
                    param_std = param_values.std().item()
                    param_min = param_values.min().item()
                    param_max = param_values.max().item()
                    param_absmax = param_values.abs().max().item()
                    print(f"[{name}] Param - Mean: {param_mean:.6f}, Std: {param_std:.6f}, "
                          f"Range: [{param_min:.6f}, {param_max:.6f}], AbsMax: {param_absmax:.6f}")
                
                # Only diagnose prediction-related NNs
                compute_stats(self.elbo_computer.state_transition.influence_nn, "InfluenceNN")
                compute_stats(self.elbo_computer.state_transition.self_nn, "SelfNN")
                
                print("=" * 50)
            except Exception as e:
                print(f"[GradDiag] Error: {e}")
        # === End diagnostics ===


        # # === 完整诊断 ===
        # print("\n" + "="*60)
        # print("GRADIENT DIAGNOSIS")
        # print("="*60)

        # # 1. 检查参数组数量
        # print(f"Number of param groups: {len(self.optimizer.param_groups)}")
        # for i, group in enumerate(self.optimizer.param_groups):
        #     print(f"  Group {i}: {len(group['params'])} parameters")

        # # 2. 裁剪前 - 计算所有参数组的范数
        # print("\n--- BEFORE CLIPPING ---")
        # all_params_list = []
        # for i, group in enumerate(self.optimizer.param_groups):
        #     group_params = group['params']
        #     all_params_list.extend(group_params)
            
        #     # 计算这个组的范数
        #     group_norm = 0.0
        #     for p in group_params:
        #         if p.grad is not None:
        #             group_norm += p.grad.data.norm(2).item() ** 2
        #     group_norm = group_norm ** 0.5
        #     print(f"  Group {i} norm: {group_norm:.2f}")

        # # 计算总范数
        # total_norm_manual = 0.0
        # for p in all_params_list:
        #     if p.grad is not None:
        #         total_norm_manual += p.grad.data.norm(2).item() ** 2
        # total_norm_manual = total_norm_manual ** 0.5
        # print(f"  TOTAL norm (manual): {total_norm_manual:.2f}")

        # # 3. 看看哪些参数梯度最大
        # print("\n--- Top 5 parameters by gradient norm ---")
        # param_norms = []
        # model_components = [
        #     ("NetworkTypeNN", self.mean_field_posterior.network_type_nn),
        #     ("SelfNN", self.elbo_computer.state_transition.self_nn),
        #     ("InfluenceNN", self.elbo_computer.state_transition.influence_nn),
        #     ("InteractionNN", self.elbo_computer.network_evolution.interaction_nn),
        # ]

        # for component_name, component in model_components:
        #     for name, param in component.named_parameters():
        #         if param.grad is not None:
        #             pnorm = param.grad.norm().item()
        #             param_norms.append((f"{component_name}.{name}", pnorm, param.shape))

        # param_norms.sort(key=lambda x: x[1], reverse=True)
        # for name, pnorm, shape in param_norms[:5]:
        #     print(f"  {name} (shape={shape}): {pnorm:.2f}")

        # # 4. 执行裁剪（只裁剪第一组）
        # print("\n--- CLIPPING (Group 0 only, max_norm=10.0) ---")
        # params_group0 = self.optimizer.param_groups[0]['params']
        # returned_norm = torch.nn.utils.clip_grad_norm_(params_group0, max_norm=10.0)
        # print(f"  Returned norm: {returned_norm:.2f}")

        # # 5. 裁剪后 - 重新计算范数
        # print("\n--- AFTER CLIPPING ---")
        # for i, group in enumerate(self.optimizer.param_groups):
        #     group_params = group['params']
        #     group_norm = 0.0
        #     for p in group_params:
        #         if p.grad is not None:
        #             group_norm += p.grad.data.norm(2).item() ** 2
        #     group_norm = group_norm ** 0.5
        #     print(f"  Group {i} norm: {group_norm:.2f}")

        # # 重新计算总范数
        # total_norm_after = 0.0
        # for p in all_params_list:
        #     if p.grad is not None:
        #         total_norm_after += p.grad.data.norm(2).item() ** 2
        # total_norm_after = total_norm_after ** 0.5
        # print(f"  TOTAL norm (after): {total_norm_after:.2f}")

        # # 6. 判断
        # print("\n--- ANALYSIS ---")
        # if len(self.optimizer.param_groups) > 1:
        #     print(f"⚠️  You have {len(self.optimizer.param_groups)} param groups!")
        #     print(f"⚠️  You're only clipping group 0, other groups are NOT clipped!")
        #     print(f"⚠️  This is why total norm is still large!")
        # elif returned_norm > 10.1:
        #     print("⚠️  WARNING: Clipping didn't work even for group 0!")
        # else:
        #     print("✓ Clipping worked correctly")

        # print("="*60 + "\n")



        # total_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=10.0)
        # print(f"Gradient norm: {total_norm:.6f}")

        print("\n=== IMMEDIATE TEST ===")
        print(f"Number of param groups: {len(self.optimizer.param_groups)}")

        # 不裁剪，只查看
        test_norm = torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'], 
            max_norm=float('inf')  # 不裁剪，只返回范数
        )
        print(f"Original norm (no clipping): {test_norm:.2f}")

    
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'], 
            max_norm=50.0
        )
        print(f"After clipping to 50.0: {clipped_norm:.2f}")

        verify_norm = torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'], 
            max_norm=float('inf')
        )
        print(f"Verify after clipping: {verify_norm:.2f}")
        print("===================\n")

        self.optimizer.step()
        print(f"Backward pass finished")
        
        # Clamp observation parameters
        with torch.no_grad():
            # self.elbo_computer.rho_1.clamp_(1e-4, 0.7)
            # self.elbo_computer.rho_2.clamp_(1e-4, 0.8)
            # self.elbo_computer.network_evolution.alpha_bonding.clamp_(1e-4, 0.1)
            # self.elbo_computer.network_evolution.beta_form.clamp_(1e-4, 0.2)
            # self.elbo_computer.network_evolution.gamma.clamp_(1e-4, 0.5)
            self.elbo_computer.network_evolution.alpha_0.clamp_(1e-4, 3)
            # self.elbo_computer.network_evolution.beta_0.clamp_(1e-4, 0.2)
            # self.elbo_computer.rho_1.clamp_(0.1, 0.7)              # MORE RESTRICTIVE RANGE
            # self.elbo_computer.rho_2.clamp_(0.2, 0.8)              # PREVENT 95% MISS RATE
            # self.elbo_computer.network_evolution.alpha_bonding.clamp_(1e-4, 5.0)   # MUCH LARGER RANGE
            # self.elbo_computer.network_evolution.beta_form.clamp_(1e-4, 10.0)      # MUCH LARGER RANGE  
            # self.elbo_computer.network_evolution.gamma.clamp_(1e-4, 0.8)          # WIDER RANGE
            # self.elbo_computer.network_evolution.alpha_0.clamp_(1e-4, 5.0)         # MUCH LARGER RANGE
            # self.elbo_computer.network_evolution.beta_0.clamp_(1e-4, 10.0)         # MUCH LA

            
        # Average metrics across batches
        num_batches = len(node_batches)
        for key in total_metrics.keys():
            total_metrics[key] /= num_batches
        
        # Return complete metrics
        metrics = {
            'epoch': self.epoch,
            'temperature': temperature,
            'num_samples': num_samples,
            'num_batches': num_batches,
            # 'rho_1': self.elbo_computer.rho_1.item(),
            # 'rho_2': self.elbo_computer.rho_2.item(),
            **total_metrics
        }
        
        self.training_history.append(metrics)
        self.epoch += 1
        
        return metrics
    
    def train_epoch_full(self,
                        features: torch.Tensor,
                        states: torch.Tensor,
                        distances: torch.Tensor,
                        network_data,
                        ground_truth_network,
                        max_timestep: int,
                        max_epochs: int,
                        lambda_constraint) -> Dict[str, float]:
        """Train one epoch with full batch (for small networks)."""
        
        # Create single batch with all nodes
        all_nodes = torch.arange(features.shape[0], dtype=torch.long)
        return self.train_epoch_batched(
            features, states, distances, network_data, ground_truth_network, [all_nodes], max_timestep, max_epochs, lambda_constraint
        )
    
    def train(self,
            features: torch.Tensor,
            states: torch.Tensor,
            distances: torch.Tensor,
            network_data,
            ground_truth_network,
            max_timestep: int,
            max_epochs: int = 1000,
            node_batches: Optional[List[torch.Tensor]] = None,
            lambda_constraint: float = 0.01,
            verbose: bool = True,
            early_stopping: bool = True,
            patience: int = 100) -> List[Dict[str, float]]:
        """
        Full training loop with updated formulation and early stopping.
        """
        use_batching = node_batches is not None
        n_households = features.shape[0]

        if not use_batching:
            if verbose:
                print(f"Training with full batch ({n_households} households)")
        else:
            if verbose:
                print(f"Training with mini-batches ({len(node_batches)} batches, "
                    f"avg {n_households / len(node_batches):.1f} households per batch)")
                
        # NEW: distance-based neighbor filter!!!!
        TOP_K = min(50, n_households-1)      # start here; tune later
        RADIUS = None   # or a float cutoff (same unit as distances)
        neighbor_index = build_neighbor_index_from_distances(
            distances, radius=RADIUS, top_k=TOP_K
        )
        network_data.neighbor_index = neighbor_index
        print(f"[Sparse] avg neighbors per node = "
            f"{sum(len(v) for v in neighbor_index)/len(neighbor_index):.2f}")

        progress_bar = tqdm(range(max_epochs), desc="Training") if verbose else range(max_epochs)

        # Early stopping variables
        best_state_likelihood = float('-inf')
        epochs_no_improve = 0

        for _ in progress_bar:
            if use_batching:
                metrics = self.train_epoch_batched(
                    features, states, distances, network_data, ground_truth_network,
                    node_batches, max_timestep, max_epochs, lambda_constraint
                )
            else:
                metrics = self.train_epoch_full(
                    features, states, distances, network_data, ground_truth_network,
                    max_timestep, max_epochs, lambda_constraint
                )
            
            if early_stopping and (self.epoch - 1) >= 650:
                current_state = metrics['state_likelihood']
                
                if current_state > best_state_likelihood + 1e-4:
                    best_state_likelihood = current_state
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"Epoch {self.epoch}: State likelihood = {current_state:.4f}, "
                      f"Best = {best_state_likelihood:.4f}, "
                      f"No improvement epochs = {epochs_no_improve}")
                
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {self.epoch}: State likelihood stopped improving")
                    break

            if verbose and self.epoch % 1 == 0:
                if hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({
                        'ELBO': f"{metrics['total_elbo']:.2f}",
                        'Temp': f"{metrics['temperature']:.3f}",
                        'Samples': metrics['num_samples'],
                        'Batches': metrics.get('num_batches', 1)
                    })

        return self.training_history
    
    