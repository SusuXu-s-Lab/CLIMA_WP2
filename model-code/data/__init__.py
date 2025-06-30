from .data_loader import DataLoader
from .synthetic_generator import DisasterRecoveryDataGenerator
from .synthetic_generator_v2 import DisasterRecoveryDataGeneratorV2

__all__ = ['DataLoader', 'DisasterRecoveryDataGenerator', 'DisasterRecoveryDataGeneratorV2']


# =============================================================================
# Usage Example

"""
Simple usage example:

from data import DataLoader

# 1. Load data
loader = DataLoader('data/', device='cpu')
data = loader.load_data()

# 2. Train/test split
train_data, test_data = loader.train_test_split(data, train_end_time=12)

# 3. Create batches
# Full batch (for small datasets)
batches = loader.create_node_batches(data['n_households'])

# Mini-batches (for larger datasets)  
batches = loader.create_node_batches(data['n_households'], batch_size=50)

# 4. Access data
states = data['states']  # [N, T, 3]
features = data['features']  # [N, F]
distances = data['distances']  # [N, N]
observed_network = data['observed_network']

# 5. Check network
is_obs = observed_network.is_observed(i=5, j=10, t=3)
link_type = observed_network.get_link_type(i=5, j=10, t=3)
hidden_pairs = observed_network.get_hidden_pairs(t=3)

# 6. Use batches
for batch_idx, node_batch in enumerate(batches):
    print(f"Batch {batch_idx}: nodes {node_batch}")
    # For this batch, you need ALL their connections to ALL other nodes
    # That's the key insight for temporal network batching
"""