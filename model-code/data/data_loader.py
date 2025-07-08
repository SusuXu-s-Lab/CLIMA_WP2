import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


class NetworkData:
    """Simple network data storage."""
    
    def __init__(self, n_households: int, n_timesteps: int):
        self.n_households = n_households
        self.n_timesteps = n_timesteps
        
        # Store observed links: {timestep: {(i,j): link_type}}
        self.observed_links = {t: {} for t in range(n_timesteps)}
        
        # All possible pairs (i < j)
        self.all_pairs = [(i, j) for i in range(n_households) for j in range(i + 1, n_households)]
    
    def add_edges(self, network_df: pd.DataFrame):
        """Add edges from network DataFrame."""
        for _, row in network_df.iterrows():
            i = int(row['household_id_1']) - 1  # Convert to 0-indexed
            j = int(row['household_id_2']) - 1
            t = int(row['timestep'])
            link_type = int(row['link_type'])
            
            # Ensure i < j
            i, j = min(i, j), max(i, j)
            self.observed_links[t][(i, j)] = link_type
    
    def is_observed(self, i: int, j: int, t: int) -> bool:
        """Check if link (i,j) is observed at time t."""
        i, j = min(i, j), max(i, j)
        return (i, j) in self.observed_links[t]
    
    def get_link_type(self, i: int, j: int, t: int) -> int:
        """Get link type if observed, else return 0."""
        i, j = min(i, j), max(i, j)
        return self.observed_links[t].get((i, j), 0)
    
    def get_hidden_pairs(self, t: int) -> List[Tuple[int, int]]:
        """Get list of hidden pairs at time t."""
        observed_pairs = set(self.observed_links[t].keys())
        return [pair for pair in self.all_pairs if pair not in observed_pairs]
    
    def get_observed_edges_at_time(self, t: int):
        """
        Return a list of (i, j, link_type) tuples for all observed edges at time t.
        """
        return [(i, j, link_type) for (i, j), link_type in self.observed_links[t].items()]


class DataLoader:
    """
    Simple data loader for disaster recovery network analysis.
    
    Does 3 things:
    1. Load CSV files and convert to tensors
    2. Train/test split
    3. Create node batches
    """
    
    def __init__(self, data_dir: str, device: str = 'cpu',
                 file_list: List[str] = ['household_states.csv', 'household_features.csv',
                                         'household_locations.csv', 'observed_network.csv',
                                         'ground_truth_network.csv']):
        self.data_dir = Path(data_dir)
        self.device = device
        self.file_list = file_list
    
    def load_data(self) -> Dict:
        """Load all data and convert to tensors."""
        
        # Load CSV files
        states_df = pd.read_csv(self.data_dir / self.file_list[0])
        features_df = pd.read_csv(self.data_dir / self.file_list[1])
        locations_df = pd.read_csv(self.data_dir / self.file_list[2])
        observed_df = pd.read_csv(self.data_dir / self.file_list[3])
        ground_truth_df = pd.read_csv(self.data_dir / self.file_list[4])
        
        # Get dimensions
        n_households = len(features_df)
        n_timesteps = len(states_df['timestep'].unique())
        
        # Normalize feature columns (excluding household_id)
        feature_cols = [col for col in features_df.columns if ((col != 'household_id') and (col != 'home'))]
        features_norm = features_df.copy()
        for col in feature_cols:
            col_values = features_df[col].values.astype(float)
            mean = col_values.mean()
            std = col_values.std()
            if std > 0:
                features_norm[col] = (col_values - mean) / std
            else:
                features_norm[col] = 0.0  # If constant column
        
        # Convert to tensors
        states = self._create_states_tensor(states_df, n_households, n_timesteps)
        features = self._create_features_tensor(features_norm, n_households)
        distances = self._create_distance_matrix(locations_df, n_households)
        
        # Create network data
        observed_network = NetworkData(n_households, n_timesteps)
        observed_network.add_edges(observed_df)
        
        ground_truth_network = NetworkData(n_households, n_timesteps)
        ground_truth_network.add_edges(ground_truth_df)
        
        # Move to device
        if self.device != 'cpu':
            states = states.to(self.device)
            features = features.to(self.device)
            distances = distances.to(self.device)
        
        return {
            'states': states,  # [N, T, 3]
            'features': features,  # [N, F]
            'distances': distances,  # [N, N]
            'observed_network': observed_network,
            'ground_truth_network': ground_truth_network,
            'n_households': n_households,
            'n_timesteps': n_timesteps,
            'feature_names': [col for col in features_df.columns if col != 'household_id']
        }
    
    def _create_states_tensor(self, states_df: pd.DataFrame, n_households: int, n_timesteps: int) -> torch.Tensor:
        """Create states tensor [N, T, 3]."""
        states = torch.zeros((n_households, n_timesteps, 3), dtype=torch.float32)
        
        for _, row in states_df.iterrows():
            i = int(row['household_id']) - 1  # Convert to 0-indexed
            t = int(row['timestep'])
            states[i, t, 0] = float(row['vacant'])
            states[i, t, 1] = float(row['repair'])
            states[i, t, 2] = float(row['sell'])
        
        return states
    
    def _create_features_tensor(self, features_df: pd.DataFrame, n_households: int) -> torch.Tensor:
        """Create features tensor [N, F]."""
        feature_cols = [col for col in features_df.columns if ((col != 'household_id') and (col != 'home'))]
        features = torch.zeros((n_households, len(feature_cols)), dtype=torch.float32)
        
        for _, row in features_df.iterrows():
            i = int(row['household_id']) - 1
            for j, col in enumerate(feature_cols):
                features[i, j] = float(row[col])
        
        return features
    
    def _create_distance_matrix(self, locations_df: pd.DataFrame, n_households: int) -> torch.Tensor:
        """Create distance matrix [N, N]."""
        locations = locations_df.sort_values('household_id')[['latitude', 'longitude']].values
        distances = torch.zeros((n_households, n_households), dtype=torch.float32)
        
        for i in range(n_households):
            for j in range(n_households):
                if i != j:
                    lat_diff = (locations[j][0] - locations[i][0]) * 111
                    lon_diff = (locations[j][1] - locations[i][1]) * 111 * np.cos(np.radians((locations[i][0] + locations[j][0]) / 2))
                    distances[i, j] = np.sqrt(lat_diff**2 + lon_diff**2)
        
        return distances
    
    def train_test_split(self, data: Dict, train_end_time: int) -> Tuple[Dict, Dict]:
        """Split data into train and test sets."""
        
        # Training data (truncate time)
        train_states = data['states'][:, :train_end_time + 1, :].clone()
        
        # Training network (filter by time)
        train_observed = NetworkData(data['n_households'], train_end_time + 1)
        for t in range(train_end_time + 1):
            train_observed.observed_links[t] = data['observed_network'].observed_links[t].copy()
        
        train_data = {
            'states': train_states,
            'features': data['features'].clone(),
            'distances': data['distances'].clone(),
            'observed_network': train_observed,
            'n_households': data['n_households'],
            'n_timesteps': train_end_time + 1,
            'feature_names': data['feature_names']
        }
        
        # Test data (keep full data for evaluation)
        test_data = data.copy()
        test_data['train_end_time'] = train_end_time
        
        return train_data, test_data
    
    def create_node_batches(self, n_households: int, batch_size: int = None) -> List[torch.Tensor]:
        """
        Create node batches.
        
        Args:
            n_households: Total number of households
            batch_size: Size of each batch. If None, return single batch with all nodes.
            
        Returns:
            List of tensor batches, each containing node indices
        """
        all_nodes = torch.arange(n_households, dtype=torch.long)
        
        if batch_size is None:
            # Single batch with all nodes
            return [all_nodes]
        
        # Split into mini-batches
        batches = []
        for i in range(0, n_households, batch_size):
            batch = all_nodes[i:i + batch_size]
            batches.append(batch)
        
        return batches
    


