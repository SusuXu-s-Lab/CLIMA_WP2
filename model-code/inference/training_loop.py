import torch
import torch.optim as optim
from typing import Dict, List, Optional
from tqdm import tqdm

from .elbo_computation import ELBOComputation
from .gumbel_softmax import GumbelSoftmaxSampler
from .variational_posterior import MeanFieldPosterior


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
                 weight_decay: float = 1e-4):
        
        self.mean_field_posterior = mean_field_posterior
        self.gumbel_sampler = gumbel_sampler
        self.elbo_computer = elbo_computer
        
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
        
    def temperature_schedule(self, epoch: int, max_epochs: int) -> float:
        progress = epoch / max_epochs
        if progress < 0.4:  
            return 2.0
        else:
            return 2.0 * (0.3 / 2.0) ** ((progress - 0.4) / 0.6)
    
    def sample_schedule(self, epoch: int, max_epochs: int) -> int:
        if epoch < max_epochs * 0.6:
            return 5 
        else:
            return 3
        

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


    def monitor_link_prediction_performance(self, marginal_probs, ground_truth_network, max_timestep):
        """
        Simple link prediction monitoring - just precision and recall.
        
        Args:
            marginal_probs: Dict[str, torch.Tensor] - marginal probabilities from variational posterior
            ground_truth_network: NetworkData object with ground truth
            max_timestep: int - maximum timestep to evaluate
        
        Returns:
            Dict with precision, recall metrics
        """
        if len(marginal_probs) == 0:
            return {'precision': 0.0, 'recall': 0.0}
        
        predicted_links = 0
        true_links = 0 
        correct_links = 0
        
        processed_pairs = set()  # To avoid double counting
        
        for pair_key, marginal_prob in marginal_probs.items():
            # Parse pair_key: "i_j_t"
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            
            i, j, t = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Skip if already processed this pair at this timestep
            pair_id = (min(i, j), max(i, j), t)
            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)
            
            # Skip if timestep is out of range
            if t > max_timestep:
                continue
            
            # Prediction: 1 if any connection (argmax > 0), 0 if no connection
            predicted_exists = 1 if torch.argmax(marginal_prob).item() > 0 else 0
            
            # Ground truth
            true_link_type = ground_truth_network.get_link_type(i, j, t)
            true_exists = 1 if true_link_type > 0 else 0
            
            # Update counters
            if predicted_exists == 1:
                predicted_links += 1
            if true_exists == 1:
                true_links += 1
            if predicted_exists == 1 and true_exists == 1:
                correct_links += 1
        
        # Calculate metrics
        precision = correct_links / predicted_links if predicted_links > 0 else 0.0
        recall = correct_links / true_links if true_links > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall
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
        if self.epoch >= 300:
            for param in self.mean_field_posterior.network_type_nn.parameters():
                param.requires_grad = False
            for param in self.elbo_computer.network_evolution.parameters():
                param.requires_grad = False
        
        
        temperature = self.temperature_schedule(self.epoch, max_epochs)
        num_samples = self.sample_schedule(self.epoch, max_epochs)
        
        # Accumulate metrics across batches
        total_metrics = {
            'total_elbo': 0.0,
            'state_likelihood': 0.0,
            'observation_likelihood': 0.0,
            'prior_likelihood': 0.0,
            'posterior_entropy': 0.0,
            'confidence_regularization': 0.0,
            'constraint_penalty': 0.0
        }
        
        self.optimizer.zero_grad()
        
        # Process each batch
        for batch_idx, node_batch in enumerate(node_batches):
            
            # Compute BOTH conditional and marginal probabilities
            conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
                    features, states, distances, node_batches[0], network_data, max_timestep)
            print(f"Conditional and marginal probabilities finished for batch {batch_idx + 1}/{len(node_batches)}")

            # Monitor link prediction performance 
            if batch_idx == 0:  # Only check first batch
                link_metrics = self.monitor_link_prediction_performance(
                    marginal_probs, ground_truth_network, max_timestep
                )
                print(f"\n=== Epoch {self.epoch} Link inference vs Ground Truth ===")
                print(f"Link Prediction - Precision: {link_metrics['precision']:.3f}, "
                    f"Recall: {link_metrics['recall']:.3f}")

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
            
            # Compute ELBO using BOTH conditional and marginal probabilities
            batch_elbo = self.elbo_computer.compute_elbo_batch(
                features, states, distances, node_batch, network_data,
                conditional_probs, marginal_probs, gumbel_samples, max_timestep, lambda_constraint, current_epoch=self.epoch
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
        
        # Update parameters after all batches
        batch_loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], float('inf'))
        print(f"Gradient norm: {total_norm:.6f}")
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

        progress_bar = tqdm(range(max_epochs), desc="Training") if verbose else range(max_epochs)

        # Early stopping variables
        best_elbo = float('-inf')
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

            # Early stopping check
            current_elbo = metrics['total_elbo']
            if early_stopping:
                if current_elbo > best_elbo + 1e-4:  # small tolerance
                    best_elbo = current_elbo
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {self.epoch} (best ELBO: {best_elbo:.2f})")
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
    
    