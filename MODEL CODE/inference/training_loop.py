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
        all_params.extend([self.elbo_computer.rho_1, self.elbo_computer.rho_2])
        
        self.optimizer = optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
        
        self.epoch = 0
        self.training_history = []
        
    def temperature_schedule(self, epoch: int, max_epochs: int) -> float:
        """Temperature scheduling."""
        progress = epoch / max_epochs
        return 2.0 * (0.1 / 2.0) ** progress
    
    def sample_schedule(self, epoch: int, max_epochs: int) -> int:
        """Sample scheduling."""
        if epoch < max_epochs * 0.2:
            return 5
        elif epoch < max_epochs * 0.6:
            return 3
        else:
            return 2
    
    def train_epoch_batched(self,
                           features: torch.Tensor,
                           states: torch.Tensor,
                           distances: torch.Tensor,
                           network_data,
                           node_batches: List[torch.Tensor],
                           max_timestep: int,
                           max_epochs: int,
                           lambda_constraint: float = 0.01) -> Dict[str, float]:
        """
        Train one epoch with updated formulation.
        """
        
        temperature = self.temperature_schedule(self.epoch, max_epochs)
        num_samples = self.sample_schedule(self.epoch, max_epochs)
        
        # Accumulate metrics across batches
        total_metrics = {
            'total_elbo': 0.0,
            'state_likelihood': 0.0,
            'observation_likelihood': 0.0,
            'prior_likelihood': 0.0,
            'posterior_entropy': 0.0,
            'sparsity_regularization': 0.0,
            'constraint_penalty': 0.0
        }
        
        self.optimizer.zero_grad()
        
        # Process each batch
        for batch_idx, node_batch in enumerate(node_batches):
            
            # Compute BOTH conditional and marginal probabilities
            conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
                features, states, distances, node_batch, network_data, max_timestep
            )
            print(f"Conditional and marginal probabilities finished for batch {batch_idx + 1}/{len(node_batches)}")
            
            # Sample hidden links using MARGINAL probabilities
            gumbel_samples = self.gumbel_sampler.sample_hidden_links_batch(
                marginal_probs, temperature, num_samples
            )
            print(f"Gumbel sampling finished for batch {batch_idx + 1}/{len(node_batches)}")
            
            # Compute ELBO using BOTH conditional and marginal probabilities
            batch_elbo = self.elbo_computer.compute_elbo_batch(
                features, states, distances, node_batch, network_data,
                conditional_probs, marginal_probs, gumbel_samples, max_timestep, lambda_constraint
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
        self.optimizer.step()
        print(f"Backward pass finished")
        
        # Clamp observation parameters
        with torch.no_grad():
            self.elbo_computer.rho_1.clamp_(1e-4, 0.7)
            self.elbo_computer.rho_2.clamp_(1e-4, 0.8)
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
            'rho_1': self.elbo_computer.rho_1.item(),
            'rho_2': self.elbo_computer.rho_2.item(),
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
                        max_timestep: int,
                        max_epochs: int,
                        lambda_constraint) -> Dict[str, float]:
        """Train one epoch with full batch (for small networks)."""
        
        # Create single batch with all nodes
        all_nodes = torch.arange(features.shape[0], dtype=torch.long)
        return self.train_epoch_batched(
            features, states, distances, network_data, [all_nodes], max_timestep, max_epochs, lambda_constraint
        )
    
    def train(self,
            features: torch.Tensor,
            states: torch.Tensor,
            distances: torch.Tensor,
            network_data,
            max_timestep: int,
            max_epochs: int = 1000,
            node_batches: Optional[List[torch.Tensor]] = None,
            lambda_constraint: float = 0.5,
            verbose: bool = True,
            early_stopping: bool = True,
            patience: int = 20) -> List[Dict[str, float]]:
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
                    features, states, distances, network_data, 
                    node_batches, max_timestep, max_epochs, lambda_constraint
                )
            else:
                metrics = self.train_epoch_full(
                    features, states, distances, network_data, 
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
    
    