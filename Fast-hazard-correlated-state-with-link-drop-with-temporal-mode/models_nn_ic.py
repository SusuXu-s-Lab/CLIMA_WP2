# models_nn_ic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim, out_dim, hidden=64, depth=2, act=nn.ReLU):
    layers, d = [], in_dim
    for _ in range(max(0, depth - 1)):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

class SelfJointMLP(nn.Module):
    """
    Joint self mechanism:
      input : (B, 2D + [1])  -> [z_i || x_i_prev || [t_norm]]
      output: (B, D)         -> per-dimension self-activation probabilities
      
      Optional time feature: t_norm ∈ [0, 1] normalized time step
    """
    def __init__(self, D, hidden=64, depth=2, use_time=False):
        super().__init__()
        self.D = D
        self.use_time = use_time
        in_dim = 2 * D + (1 if use_time else 0)
        self.net = mlp(in_dim=in_dim, out_dim=D, hidden=hidden, depth=depth)

    def forward(self, z_i, x_i_prev, t_norm=None):
        # z_i, x_i_prev: (B, D)
        # t_norm: (B, 1) or None
        if self.use_time:
            if t_norm is None:
                raise ValueError("t_norm must be provided when use_time=True")
            h = torch.cat([z_i, x_i_prev, t_norm], dim=-1)
        else:
            h = torch.cat([z_i, x_i_prev], dim=-1)
        logits = self.net(h)
        return torch.sigmoid(logits)  # (B, D)

class EdgeJointMonotone(nn.Module):
    """
    Joint edge mechanism (per attempt):
      Inputs:
        - phi_ji:   (B,)        edge influence scalar (standardized internally)
        - k:        (B,) long   source dimension newly active
        - z_i,x_i:  (B,D)       dest node context at t-1
        - z_j,x_j:  (B,D)       source node context at t-1
        - both_d1:  (B,1)       1.0 if x_i_prev[d1]==1 and x_j_prev[d1]==1 else 0.0
        - t_norm:   (B,1)       [optional] normalized time ∈ [0,1]
      Output:
        - (B, D): per-destination-dim probabilities for this attempt
      Monotonic in phi via:
        logit_d = ctx_d + softplus(a_d) * phi_std
    """
    def __init__(self, D, emb_dim=8, hidden=96, depth=3, phi_monotone=True, include_drop_flag=True, use_time=False):
        super().__init__()
        self.D = D
        self.phi_monotone = phi_monotone
        self.include_drop_flag = include_drop_flag
        self.use_time = use_time

        self.emb_k = nn.Embedding(D, emb_dim)

        # running stats for phi standardization
        self.phi_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.phi_var  = nn.Parameter(torch.ones(1),  requires_grad=False)

        # context net: emb(k) + z_i + x_i + z_j + x_j + [both_d1] + [t_norm]
        extra = (1 if include_drop_flag else 0) + (1 if use_time else 0)
        ctx_dim = (emb_dim + 4*D) + extra
        self.ctx_net = mlp(ctx_dim, out_dim=D, hidden=hidden, depth=depth)

        if phi_monotone:
            self.w_phi = nn.Parameter(torch.zeros(D))  # one slope per dest dim (softplus >= 0)
        else:
            self.w_phi = None  # (not recommended)

    def forward(self, phi, k, z_i, x_i, z_j, x_j, both_d1=None, t_norm=None, update_stats=False):
        """
        phi: (B,) float; k: (B,) long; z_i,x_i,z_j,x_j: (B,D) float
        both_d1: (B,1) float in {0,1}; can be None if include_drop_flag=False
        t_norm: (B,1) float in [0,1]; can be None if use_time=False
        returns: (B,D) probabilities
        """
        if update_stats:
            with torch.no_grad():
                m = phi.mean()
                v = phi.var(unbiased=False).clamp_min(1e-6)
                self.phi_mean.copy_(m.view_as(self.phi_mean))
                self.phi_var.copy_(v.view_as(self.phi_var))

        phi_std = (phi - self.phi_mean) / torch.sqrt(self.phi_var + 1e-8)

        parts = [self.emb_k(k), z_i, x_i, z_j, x_j]
        if self.include_drop_flag:
            if both_d1 is None:
                raise ValueError("both_d1 must be provided when include_drop_flag=True")
            parts.append(both_d1)  # (B,1)
        if self.use_time:
            if t_norm is None:
                raise ValueError("t_norm must be provided when use_time=True")
            parts.append(t_norm)  # (B,1)
        ctx = torch.cat(parts, dim=-1)  # (B, ctx_dim)
        h = self.ctx_net(ctx)           # (B, D)

        if self.phi_monotone:
            slope = F.softplus(self.w_phi).unsqueeze(0)   # (1, D) >= 0
            logits = h + slope * phi_std.unsqueeze(-1)    # (B, D)
        else:
            logits = h

        return torch.sigmoid(logits)
