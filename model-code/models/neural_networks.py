import torch
import torch.nn as nn
from .utils import compute_pairwise_features


class SeqSelfNN(nn.Module):
    """
    Self-activation channel: seq2seq
    Input:  [N, T, 3+feature_dim] - [s_i(t), x_i(static)]
    Output: [N, T, 3] - hazard for each decision type
    """
    def __init__(self, feature_dim: int = None, hidden_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        if feature_dim is not None:
            self.gru = nn.GRU(feature_dim + 3, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 3)
        else:
            self.gru = None
            self.head = None
    
    def _lazy_init(self, input_dim: int):
        """Initialize layers on first forward pass"""
        self.feature_dim = input_dim - 3
        self.gru = nn.GRU(input_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, 3)
        
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self.gru = self.gru.to(device)
            self.head = self.head.to(device)

    def forward(self, x_seq):
        """
        Args:
            x_seq: [N, T, 3+feature_dim]
        Returns:
            h_seq: [N, T, 3] - hazard probabilities
            logits: [N, T, 3] - raw logits
        """
        if self.gru is None:
            self._lazy_init(x_seq.shape[-1])
        
        y_all, _ = self.gru(x_seq)
        logits = self.head(y_all)
        h_seq = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        return h_seq, logits


class SeqPairInfluenceNN(nn.Module):
    """
    Neighbor influence channel: seq2seq (shared across all pairs)
    Input:  [P, T, 10+pairwise_feat_dim] - [g_ij(t), s_i(t), s_j(t), Δx_ij, d_ij]
    Output: [P, T, 3] - hazard for each decision type
    """
    def __init__(self, pairwise_feature_dim: int = None, hidden_dim: int = 32):
        super().__init__()
        self.pairwise_feature_dim = pairwise_feature_dim
        self.hidden_dim = hidden_dim
        
        if pairwise_feature_dim is not None:
            input_dim = 3 + 3 + 3 + pairwise_feature_dim + 1
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 3)
        else:
            self.gru = None
            self.head = None
    
    def _lazy_init(self, input_dim: int):
        """Initialize layers on first forward pass"""
        self.pairwise_feature_dim = input_dim - 10
        self.gru = nn.GRU(input_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, 3)
        
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self.gru = self.gru.to(device)
            self.head = self.head.to(device)

    def forward(self, x_seq):
        """
        Args:
            x_seq: [P, T, 10+pairwise_feat_dim]
        Returns:
            h_seq: [P, T, 3] - hazard probabilities
            logits: [P, T, 3] - raw logits
        """
        if self.gru is None:
            self._lazy_init(x_seq.shape[-1])
        
        y_all, _ = self.gru(x_seq)
        logits = self.head(y_all)
        h_seq = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        return h_seq, logits


class NetworkTypeNN(nn.Module):
    """Network type posterior: [f_ij, S_i, S_j, prev_link_type, dist, is_initial] -> [3]"""
    
    def __init__(self, feature_dim: int, L: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.L = L
        input_dim = feature_dim + L * 3 + L * 3 + 3 + 1 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, features_i, features_j, state_history_i, state_history_j, 
                prev_link_type, distances, is_initial):
        f_ij = compute_pairwise_features(features_i, features_j)
        x = torch.cat([f_ij, state_history_i, state_history_j, prev_link_type, distances, is_initial], dim=1)
        return self.network(x)


class InteractionFormationNN(nn.Module):
    """Network formation prior: [f_ij, s_i, s_j, dist] -> [1]"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        input_dim = feature_dim + 3 + 3 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features_i, features_j, states_i, states_j, distances):
        f_ij = compute_pairwise_features(features_i, features_j)
        x = torch.cat([f_ij, states_i, states_j, distances], dim=1)
        return torch.sigmoid(self.network(x))