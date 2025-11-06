# em_trainer_nn_ic.py
# EM trainer for Neural IC with D joint states and dynamic adjacency At:
# Adds:
#   - both_d1 flag into edge inputs (explicitly models drop-on-d1)
#   - responsibility annealing (tau, edge_resp_floor schedules)
#   - optional phi-stats warmup (stabilizes early edge training)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from models_nn_ic import SelfJointMLP, EdgeJointMonotone

# -------- dataset construction --------

def build_dataset(sim: Dict) -> List[Dict]:
    """
    Build per-(t,i,d) records with attempts from neighbors newly active at t-1.
    Uses dynamic adjacency At[t-1] if provided; else falls back to static A.
    Adds 'both_d1' attempt flag when sim['true_params']['drop_dim']>=0.
    """
    A = sim.get("A", None)
    At = sim.get("At", None)
    X, Y, z, phi = sim["X"], sim["Y"], sim["z"], sim["phi"]
    T, N, D = X.shape
    drop_dim = sim.get("true_params", {}).get("drop_dim", -1)

    data = []
    for t in range(1, T):
        newly = Y[t-1]
        A_used = At[t-1] if At is not None else A
        for i in range(N):
            z_i = z[i].astype(np.float32)
            x_i = X[t-1, i].astype(np.float32)
            js = np.where(A_used[:, i] == 1)[0]
            for d in range(D):
                if X[t-1, i, d] == 1:
                    continue
                y = int(Y[t, i, d])
                attempts = []
                for j in js:
                    if not newly[j].any():
                        continue
                    x_j_prev = X[t-1, j].astype(np.float32)
                    z_j = z[j].astype(np.float32)
                    # per-dimension attempts from j that newly activated
                    for k in range(D):
                        if newly[j, k] == 1:
                            both_d1 = 0.0
                            if isinstance(drop_dim, int) and (0 <= drop_dim < D):
                                both_d1 = float((x_i[drop_dim] == 1.0) and (x_j_prev[drop_dim] == 1.0))
                            attempts.append(dict(
                                phi=float(phi[j, i]),
                                k=int(k),
                                z_j=z_j,
                                x_j=x_j_prev,
                                both_d1=both_d1
                            ))
                data.append(dict(
                    y=y, i=int(i), d=int(d), t=int(t),
                    z_i=z_i, x_i=x_i, attempts=attempts
                ))
    return data

# -------- E step helpers --------

@torch.no_grad()
def _noisyor_p(s, e_list):
    if len(e_list) == 0: return s
    prod = 1.0
    for e in e_list: prod *= (1.0 - e)
    return 1.0 - (1.0 - s) * prod

def _sharpen_and_normalize(r_self: float, r_edges: list, tau: float):
    if tau is None or abs(tau - 1.0) < 1e-8:
        return r_self, r_edges
    r_s = r_self ** (1.0 / tau)
    r_e = [re ** (1.0 / tau) for re in r_edges]
    s = r_s + sum(r_e) + 1e-12
    return r_s / s, [re / s for re in r_e]

def e_step(data, self_net, edge_net, tau_resp=1.0, edge_resp_floor=0.0, device="cpu"):
    """
    Returns:
      self_rows: (z_i, x_i, d, target) for the self BCE
      edge_rows: (phi, k, z_i, x_i, z_j, x_j, both_d1, d, target) for the edge BCE
    """
    self_rows = []
    edge_rows = []
    for rec in data:
        y = rec["y"]; d = rec["d"]
        z_i = torch.tensor(rec["z_i"], dtype=torch.float32, device=device).unsqueeze(0)
        x_i = torch.tensor(rec["x_i"], dtype=torch.float32, device=device).unsqueeze(0)

        s_vec = self_net(z_i, x_i).squeeze(0)  # (D,)
        s = float(s_vec[d].item())

        e_vals = []
        both_flags = []
        if rec["attempts"]:
            phi = torch.tensor([a["phi"] for a in rec["attempts"]], dtype=torch.float32, device=device)
            k   = torch.tensor([a["k"]   for a in rec["attempts"]], dtype=torch.long, device=device)
            z_j = torch.tensor([a["z_j"] for a in rec["attempts"]], dtype=torch.float32, device=device)
            x_j = torch.tensor([a["x_j"] for a in rec["attempts"]], dtype=torch.float32, device=device)
            both = torch.tensor([[a["both_d1"]] for a in rec["attempts"]], dtype=torch.float32, device=device)  # (M,1)
            z_i_rep = z_i.repeat(len(rec["attempts"]), 1)
            x_i_rep = x_i.repeat(len(rec["attempts"]), 1)
            q_mat = edge_net(phi, k, z_i_rep, x_i_rep, z_j, x_j, both, update_stats=False)  # (M,D)
            e_vals = q_mat[:, d].detach().cpu().numpy().tolist()
            both_flags = both.detach().cpu().numpy().tolist()

        if y == 0:
            self_rows.append((rec["z_i"], rec["x_i"], d, 0.0))
            for idx, a in enumerate(rec["attempts"]):
                edge_rows.append((a["phi"], a["k"], rec["z_i"], rec["x_i"], a["z_j"], a["x_j"], a["both_d1"], d, 0.0))
            continue

        # y=1: responsibilities (noisy-OR)
        prodE = 1.0
        for e in e_vals: prodE *= (1.0 - e)
        p = 1.0 - (1.0 - s) * prodE
        denom = max(p, 1e-12)

        r_self = (s * prodE) / denom
        r_edges = []
        for m, e_m in enumerate(e_vals):
            prod_o = 1.0
            for j, e_j in enumerate(e_vals):
                if j == m: continue
                prod_o *= (1.0 - e_j)
            r_m = ((1.0 - s) * e_m * prod_o) / denom
            r_edges.append(r_m)

        r_self, r_edges = _sharpen_and_normalize(r_self, r_edges, tau_resp)

        if edge_resp_floor and r_edges:
            tot = r_self + sum(r_edges) + 1e-12
            r_self = r_self / tot
            r_edges = [r / tot for r in r_edges]
            edge_sum = sum(r_edges)
            if edge_sum < edge_resp_floor:
                if edge_sum <= 1e-12:
                    r_edges = [edge_resp_floor / len(r_edges)] * len(r_edges)
                else:
                    scale = edge_resp_floor / edge_sum
                    r_edges = [r * scale for r in r_edges]
                r_self = max(1e-6, 1.0 - sum(r_edges))

        self_rows.append((rec["z_i"], rec["x_i"], d, float(r_self)))
        for (a, r_m) in zip(rec["attempts"], r_edges):
            edge_rows.append((a["phi"], a["k"], rec["z_i"], rec["x_i"], a["z_j"], a["x_j"], a["both_d1"], d, float(r_m)))

    return self_rows, edge_rows

# -------- losses --------

def _compute_dim_weights_self(self_rows, D, mode="none", device="cpu"):
    if mode == "none":
        return torch.ones(D, dtype=torch.float32, device=device)

    sums = np.zeros(D, dtype=np.float64)
    cnts = np.zeros(D, dtype=np.float64)
    for _, _, d, y in self_rows:
        sums[d] += float(y)
        cnts[d] += 1.0
    prev = sums / np.maximum(cnts, 1e-12)

    if mode == "invfreq":
        w = 1.0 / np.clip(prev, 1e-4, None)
    elif mode == "sqrt_invfreq":
        w = 1.0 / np.sqrt(np.clip(prev, 1e-6, None))
    else:
        w = np.ones(D, dtype=np.float64)

    sample_counts = cnts
    mean_w = (w * sample_counts).sum() / max(sample_counts.sum(), 1.0)
    w_norm = w / mean_w if mean_w > 0 else w
    return torch.tensor(w_norm, dtype=torch.float32, device=device)

def _bce(pred, target):
    pred = torch.clamp(pred, 1e-6, 1-1e-6)
    target = target.float()
    return F.binary_cross_entropy(pred, target, reduction="mean")

def _self_loss(rows, self_net, device, dim_weights=None):
    if not rows: return torch.tensor(0.0, device=device)
    z = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=device)  # (B,D)
    x = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=device)  # (B,D)
    d = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)     # (B,)
    y = torch.tensor([r[3] for r in rows], dtype=torch.float32, device=device)  # (B,)
    s_mat = self_net(z, x)                                                      # (B,D)
    s = s_mat[torch.arange(d.size(0), device=device), d]                        # (B,)

    if dim_weights is None:
        return _bce(s, y)

    per = F.binary_cross_entropy(torch.clamp(s, 1e-6, 1-1e-6), y, reduction="none")
    w = dim_weights[d]  # (B,)
    return (per * w).mean()

def _edge_loss(rows, edge_net, device):
    if not rows: return torch.tensor(0.0, device=device)
    phi = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=device)
    k   = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    z_i = torch.tensor([r[2] for r in rows], dtype=torch.float32, device=device)
    x_i = torch.tensor([r[3] for r in rows], dtype=torch.float32, device=device)
    z_j = torch.tensor([r[4] for r in rows], dtype=torch.float32, device=device)
    x_j = torch.tensor([r[5] for r in rows], dtype=torch.float32, device=device)
    both = torch.tensor([[r[6]] for r in rows], dtype=torch.float32, device=device)  # (B,1)
    d   = torch.tensor([r[7] for r in rows], dtype=torch.long, device=device)
    y   = torch.tensor([r[8] for r in rows], dtype=torch.float32, device=device)
    q_mat = edge_net(phi, k, z_i, x_i, z_j, x_j, both, update_stats=False)           # (B,D)
    q = q_mat[torch.arange(d.size(0), device=device), d]                             # (B,)
    return _bce(q, y)

def _edge_pairwise_rank_loss(rows, edge_net, device, num_pairs=512, margin=0.02):
    from collections import defaultdict
    import random
    if not rows:
        return torch.tensor(0.0, device=device)

    buckets = defaultdict(list)  # (d,k) -> list of (idx, phi)
    for idx, r in enumerate(rows):
        phi, k, _, _, _, _, _, d, _ = r
        buckets[(int(d), int(k))].append((idx, float(phi)))

    pairs = []
    keys = list(buckets.keys())
    if not keys:
        return torch.tensor(0.0, device=device)

    per_bucket = max(1, num_pairs // max(1, len(keys)))
    for dk in keys:
        items = buckets[dk]
        if len(items) < 2: continue
        tries = 0
        while len(pairs) < num_pairs and tries < per_bucket * 4:
            i1, i2 = random.sample(items, 2); tries += 1
            if i1[1] == i2[1]: continue
            lo, hi = (i1, i2) if i1[1] < i2[1] else (i2, i1)
            pairs.append((lo[0], hi[0], dk[0], dk[1]))
        if len(pairs) >= num_pairs: break
    if not pairs:
        return torch.tensor(0.0, device=device)

    idx_lo = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    idx_hi = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    d      = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=device)
    k      = torch.tensor([p[3] for p in pairs], dtype=torch.long, device=device)

    def gather(col):
        return torch.tensor([rows[i][col] for i in idx_lo.tolist()], dtype=torch.float32, device=device), \
               torch.tensor([rows[i][col] for i in idx_hi.tolist()], dtype=torch.float32, device=device)

    phi_lo, phi_hi = gather(0)
    z_i_lo, z_i_hi = gather(2)
    x_i_lo, x_i_hi = gather(3)
    z_j_lo, z_j_hi = gather(4)
    x_j_lo, x_j_hi = gather(5)
    both_lo = torch.tensor([[rows[i][6]] for i in idx_lo.tolist()], dtype=torch.float32, device=device)
    both_hi = torch.tensor([[rows[i][6]] for i in idx_hi.tolist()], dtype=torch.float32, device=device)

    q_lo_mat = edge_net(phi_lo, k, z_i_lo, x_i_lo, z_j_lo, x_j_lo, both_lo, update_stats=False)  # (B,D)
    q_hi_mat = edge_net(phi_hi, k, z_i_hi, x_i_hi, z_j_hi, x_j_hi, both_hi, update_stats=False)  # (B,D)

    q_lo = q_lo_mat[torch.arange(d.size(0), device=device), d]
    q_hi = q_hi_mat[torch.arange(d.size(0), device=device), d]

    return torch.clamp(margin - (q_hi - q_lo), min=0).mean()

# -------- training loop with annealing --------

def _interp(a, b, frac):
    return a + (b - a) * frac

def train_em(sim: Dict,
             D=3,
             self_hidden=64, self_depth=2,
             edge_emb=8, edge_hidden=96, edge_depth=3, phi_monotone=True,
             em_iters=20, epochs_self=10, epochs_edge=8,
             lr_self=3e-3, lr_edge=8e-3, wd_self=1e-4, wd_edge=3e-3,
             device="cpu", verbose=True,
             # fixed (legacy) settings
             tau_resp=0.85, edge_resp_floor=0.12,
             # annealing (optional): if provided, overrides fixed per-iter
             tau_start: Optional[float]=None, tau_end: Optional[float]=None,
             edge_floor_start: Optional[float]=None, edge_floor_end: Optional[float]=None,
             # φ stats warmup
             phi_warmup_iters: int = 0,
             freeze_phi_stats: bool = True,
             # ranking regularizer
             lambda_rank=0.0, rank_pairs=512, rank_margin=0.02,
             # self weighting
             self_weighting="none"):
    torch.manual_seed(0)
    data = build_dataset(sim)

    self_net = SelfJointMLP(D, hidden=self_hidden, depth=self_depth).to(device)
    edge_net = EdgeJointMonotone(D, emb_dim=edge_emb, hidden=edge_hidden, depth=edge_depth,
                                 phi_monotone=phi_monotone, include_drop_flag=True).to(device)

    # initialize phi stats from initial edges (static A) if present
    if freeze_phi_stats:
        A = sim.get("A", None)
        At = sim.get("At", None)
        if At is not None:
            vals = torch.tensor(sim["phi"][At[0] == 1].astype(np.float32))
        elif A is not None:
            vals = torch.tensor(sim["phi"][A == 1].astype(np.float32))
        else:
            vals = torch.tensor([])
        if vals.numel() > 0:
            edge_net.phi_mean.copy_(vals.mean().view_as(edge_net.phi_mean))
            edge_net.phi_var.copy_(vals.var(unbiased=False).clamp_min(1e-6).view_as(edge_net.phi_var))

    opt_s = optim.Adam(self_net.parameters(), lr=lr_self, weight_decay=wd_self)
    opt_e = optim.Adam(edge_net.parameters(), lr=lr_edge, weight_decay=wd_edge)

    for it in range(em_iters):
        # responsibility annealing schedule
        if (tau_start is not None) and (tau_end is not None):
            frac = it / max(em_iters - 1, 1)
            tau_curr = _interp(tau_start, tau_end, frac)
        else:
            tau_curr = tau_resp

        if (edge_floor_start is not None) and (edge_floor_end is not None):
            frac = it / max(em_iters - 1, 1)
            floor_curr = _interp(edge_floor_start, edge_floor_end, frac)
        else:
            floor_curr = edge_resp_floor

        # optional: refresh phi mean/var during warmup from current rows
        if phi_warmup_iters > 0 and it < phi_warmup_iters:
            # collect all phi from attempts at this pass' responsibilities computation
            # (we momentarily compute responsibilities with previous params to harvest rows)
            tmp_self_rows, tmp_edge_rows = e_step(
                data, self_net, edge_net, tau_resp=tau_curr, edge_resp_floor=floor_curr, device=device
            )
            if tmp_edge_rows:
                phis = torch.tensor([r[0] for r in tmp_edge_rows], dtype=torch.float32)
                edge_net.phi_mean.copy_(phis.mean().view_as(edge_net.phi_mean))
                edge_net.phi_var.copy_(phis.var(unbiased=False).clamp_min(1e-6).view_as(edge_net.phi_var))

        # E step
        self_rows, edge_rows = e_step(
            data, self_net, edge_net, tau_resp=tau_curr, edge_resp_floor=floor_curr, device=device
        )

        # per-state weights for self
        dim_weights = _compute_dim_weights_self(self_rows, D, mode=self_weighting, device=device)

        # M-step: self
        for _ in range(epochs_self):
            opt_s.zero_grad()
            loss_s = _self_loss(self_rows, self_net, device, dim_weights=dim_weights)
            loss_s.backward()
            nn.utils.clip_grad_norm_(self_net.parameters(), 5.0)
            opt_s.step()

        # M-step: edge
        for _ in range(epochs_edge):
            opt_e.zero_grad()
            loss_e = _edge_loss(edge_rows, edge_net, device)
            if edge_net.w_phi is not None:
                slope = F.softplus(edge_net.w_phi).mean()
                loss_e = loss_e + 1e-3 * (1.0 / (slope + 1e-6))
            if lambda_rank and lambda_rank > 0:
                rloss = _edge_pairwise_rank_loss(edge_rows, edge_net, device,
                                                 num_pairs=rank_pairs, margin=rank_margin)
                loss_e = loss_e + lambda_rank * rloss
            loss_e.backward()
            nn.utils.clip_grad_norm_(edge_net.parameters(), 5.0)
            opt_e.step()

        if verbose:
            print(f"[EM {it+1:02d}] tau={tau_curr:.3f} floor={floor_curr:.3f} "
                  f"self_loss={loss_s.item():.4f} edge_loss={loss_e.item():.4f}")

    return self_net, edge_net
