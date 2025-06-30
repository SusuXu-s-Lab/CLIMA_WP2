from .variational_posterior import MeanFieldPosterior
from .gumbel_softmax import GumbelSoftmaxSampler
from .elbo_computation import ELBOComputation
from .training_loop import NetworkStateTrainer
# from .elbo_computation_temporal_weighted import ELBOComputationTemporalWeighted

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
}