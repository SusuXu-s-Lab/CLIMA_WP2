from .variational_posterior import MeanFieldPosterior
from .gumbel_softmax import GumbelSoftmaxSampler
from .elbo_computation import ELBOComputation
from .training_loop import NetworkStateTrainer

__all__ = [
    'MeanFieldPosterior', 'GumbelSoftmaxSampler', 'ELBOComputation', 'NetworkStateTrainer'
]

# Training configuration
TRAINING_CONFIG = {
    'gumbel_temperature_start': 2.0,
    'gumbel_temperature_end': 0.1,
    'num_samples_start': 5,
    'num_samples_end': 2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,  # Built-in L2 regularization
    'confidence_weight': 0.5,  # For entropy regularization (optional)
    'max_epochs': 1000,
    'rho_1': 0.5,  # Initial rho_1 for ELBO computation
    'rho_2': 0.5,  # Initial rho_2 for ELBO computation
    # 'L_linger': 3,
    # 'decay_type': 'exponential',  # 'linear' or 'exponential
    # 'decay_rate': 0.5,
    'max_neighbor_influences': 5  # FIXED: Was 100, causing train-eval inconsistency
}