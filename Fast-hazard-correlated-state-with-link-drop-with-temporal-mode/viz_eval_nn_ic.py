# viz_eval_nn_ic.py
# Visualize + evaluate Neural IC + EM (joint self & joint edge; dynamic At with link drop on d1)
import os, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

from em_trainer_nn_ic import train_em
from models_nn_ic import SelfJointMLP, EdgeJointMonotone

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def time_bias(t, K, big=10.0, mode="hard"):
    """
    Logit bias for slow starts (warmup mode):
      - 'hard': b(t) = -big for t < K, else 0
      - 'linear': b(t) ramps from -big at t=0 to 0 at t=K, then 0
    """
    if K is None or K <= 0:
        return 0.0
    if mode == "hard":
        return -big if t < K else 0.0
    # linear ramp
    if t >= K:
        return 0.0
    frac = 1.0 - (t / max(K, 1))
    return -big * frac


def inverse_u_with_baseline(t, start, peak, end, baseline=0.05):
    """
    Inverse-U shaped multiplier for time-varying hazards.
    Returns value in [baseline, 1.0]:
      - Ramps up from baseline to 1.0 between start and peak
      - Ramps down from 1.0 to baseline between peak and end
      - Returns baseline outside [start, end]
    
    Handles edge cases:
      - If start < 0, clips to 0
      - If end > T, that's fine (will just return baseline after T)
    """
    # Clip start to non-negative
    start = max(0.0, start)
    
    if t < start or t > end:
        return baseline
    elif t < peak:
        # Ascending phase
        progress = (t - start) / max(peak - start, 1e-8)
        return baseline + (1.0 - baseline) * progress
    else:
        # Descending phase
        progress = (t - peak) / max(end - peak, 1e-8)
        return 1.0 - (1.0 - baseline) * progress


def multiplier_to_logit_offset(m, eps=1e-8, max_suppression=10.0):
    """
    Convert inverse-U multiplier to logit-space offset (SUPPRESSION ONLY).
    
    Philosophy: Match warmup behavior - only suppress, never promote.
    - m = 1.0 (peak):     offset = 0.0 (baseline probability)
    - m = baseline (off-peak): offset = -max_suppression (strong suppression)
    
    This creates smooth inverse-U activation curves without explosive cascades.
    The offset is linear in (1-m) for interpretability:
        offset = -max_suppression * (1 - m)
    
    Args:
        m: multiplier in [baseline, 1.0]
        max_suppression: maximum suppression strength (logit units)
    """
    m = np.clip(m, eps, 1.0)
    # Linear suppression: 0 at peak (m=1), -max_suppression at baseline
    # This is analogous to warmup's time_bias ramping from -big to 0
    return -max_suppression * (1.0 - m)

def simulate_icD_dynamic(
    N=60, T=150, D=3, p_edge=0.10, p_seed=0.02,
    c0=-4.0, c1=2.0, seed=7, rho=0.0, drop_dim=1,
    a0=(-3.0, -3.3, -2.7), a1=(1.2, 0.9, 1.5), B=None,
    # Influence window parameter
    m=1,                # max influence window: nodes active in past m steps can influence
    # Time modulation mode
    time_mode="warmup",  # 'warmup' or 'inverse_u'
    # Warmup mode parameters (legacy, backward compatible)
    K_self=0,           # no self gating by default
    K_edge=0,           # no edge gating by default
    bias_self=10.0,     # strength of early suppression (logit units)
    bias_edge=10.0,
    mode_self="hard",   # 'hard' or 'linear'
    mode_edge="hard",
    # Inverse-U mode parameters
    peak_times=None,    # per-dimension peak times (D-length array)
    widths=None,        # per-dimension widths (D-length array)
    baseline=0.05,      # minimum multiplier outside active period
    max_suppression=5.0 # maximum suppression strength for inverse-U (logit units)
):
    """
    IC simulator with two time modulation modes and flexible influence window.
    
    Parameters:
    -----------
    m : int, default=1
        Maximum influence window. At time t, a source node j active in any of the 
        past m steps (i.e., activated at t-m, ..., t-1) can attempt to influence node i.
        - m=1: classic IC (only t-1 newly activated nodes can influence)
        - m>1: extended influence window (accumulates multiple exposures)
    
    time_mode : str, 'warmup' or 'inverse_u'
        1. 'warmup' mode (default, backward compatible):
           - Uses K_self/K_edge with bias_self/bias_edge
           - Suppresses activations before K steps
        
        2. 'inverse_u' mode:
           - Uses peak_times, widths, and baseline per dimension
           - Hazards follow inverse-U shape over time
           - Applied in logit space to preserve attribution
    
    Returns: dict with A, z, phi, X, Y, s_true, true_params (includes m)
    """
    rng = np.random.default_rng(seed)
    m = max(1, int(m))  # ensure m >= 1

    # Directed adjacency & arc scalar feature
    A = (rng.random((N, N)) < p_edge).astype(int)
    np.fill_diagonal(A, 0)

    phi = rng.normal(0, 1, size=(N, N)) * A

    # Correlated node features → base self probabilities (time-invariant)
    if rho is None: 
        rho = 0.0
    cov = (1 - rho) * np.eye(D) + rho * np.ones((D, D))
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(N, D)) @ L.T

    a0 = np.array(a0, dtype=float)[:D]
    a1 = np.array(a1, dtype=float)[:D]
    if B is None:
        B = np.array([[ 0.5, -0.2,  0.1],
                      [ 0.3,  0.6, -0.1],
                      [-0.1,  0.4,  0.8]], float)[:D, :D]
    else:
        B = np.array(B, float)[:D, :D]

    s_base = sigmoid(a0[None, :] + a1[None, :] * z)   # (N,D)
    s_base_logit = logit(s_base)

    # Prepare time modulation parameters
    if time_mode == "inverse_u":
        if peak_times is None:
            peak_times = np.array([T//4, T//3, T//2], dtype=float)[:D]
        else:
            peak_times = np.array(peak_times, dtype=float)[:D]
        
        if widths is None:
            widths = np.array([T//3, T//2.5, T//2], dtype=float)[:D]
        else:
            widths = np.array(widths, dtype=float)[:D]

    X = np.zeros((T, N, D), dtype=int)  # cumulative
    Y = np.zeros((T, N, D), dtype=int)  # increments
    X[0] = (rng.random((N, D)) < p_seed).astype(int)

    # We evolve A dynamically (drop on drop_dim), but return A0 to the trainer
    # so attempts are enumerated consistently; Y carries the actual dynamics.
    A0 = A.copy()

    for t in range(1, T):
        # Influence window: nodes activated in [t-m, t-1]
        # For m=1 (default), this is just Y[t-1] (classic IC)
        # For m>1, we look back m steps
        window_start = max(0, t - m)
        if m == 1:
            # Fast path for classic IC (backward compatible)
            recently_active = Y[t - 1]  # (N, D)
        else:
            # Extended window: activated any time in [window_start, t-1]
            # recently_active[j, k] = 1 if j activated on dim k in past m steps
            recently_active = np.zeros((N, D), dtype=int)
            for tau in range(window_start, t):
                recently_active = np.maximum(recently_active, Y[tau])
        
        # Compute time modulation per dimension
        if time_mode == "warmup":
            # Legacy warmup mode: single bias for self and edge
            b_s = time_bias(t, K_self, big=bias_self, mode=mode_self)
            b_e = time_bias(t, K_edge, big=bias_edge, mode=mode_edge)
            logit_offsets = np.array([b_s] * D)  # same for all dims (self)
            logit_offsets_edge = np.array([b_e] * D)  # same for all dims (edge)
        
        elif time_mode == "inverse_u":
            # Inverse-U mode: per-dimension logit offsets
            logit_offsets = np.zeros(D)
            for d in range(D):
                start_d = peak_times[d] - widths[d] / 2
                end_d = peak_times[d] + widths[d] / 2
                m_d = inverse_u_with_baseline(t, start_d, peak_times[d], end_d, baseline)
                logit_offsets[d] = multiplier_to_logit_offset(m_d, max_suppression=max_suppression)
            logit_offsets_edge = logit_offsets  # same for edge
        
        else:
            raise ValueError(f"Unknown time_mode: {time_mode}")

        for i in range(N):
            for d in range(D):
                if X[t - 1, i, d] == 1:
                    continue
                
                # Time-modulated self probability
                s_t = sigmoid(s_base_logit[i, d] + logit_offsets[d])

                # Time-modulated edge attempts from neighbors active in past m steps
                net_hit = False
                if np.any(recently_active):
                    js = np.where(A[:, i] == 1)[0]
                    for j in js:
                        for k in range(D):
                            if recently_active[j, k] == 1:
                                q_logit = c0 + c1 * phi[j, i] + B[d, k] + logit_offsets_edge[d]
                                q_t = sigmoid(q_logit)
                                if rng.random() < q_t:
                                    net_hit = True
                                    break
                        if net_hit:
                            break

                Y[t, i, d] = int((rng.random() < s_t) or net_hit)

        X[t] = np.maximum(X[t - 1], Y[t])

        # Dynamic link deletion (undirected) for next step if desired
        if drop_dim is not None and 0 <= int(drop_dim) < D:
            idx = np.where(X[t, :, int(drop_dim)] == 1)[0]
            if idx.size > 0:
                A[np.ix_(idx, idx)] = 0

    # Return A=A0 (static view for the trainer), plus all other fields
    return dict(
        A=A0, z=z, phi=phi, X=X, Y=Y, s_true=s_base,
        true_params=dict(
            c0=c0, c1=c1, B=B,
            m=m,  # influence window size
            time_mode=time_mode,
            K_self=K_self, K_edge=K_edge,
            bias_self=bias_self, bias_edge=bias_edge,
            mode_self=mode_self, mode_edge=mode_edge,
            peak_times=peak_times if time_mode == "inverse_u" else None,
            widths=widths if time_mode == "inverse_u" else None,
            baseline=baseline if time_mode == "inverse_u" else None
        )
    )



# ---- metrics (same as before) ----

def rmse(a, b): a, b = np.asarray(a), np.asarray(b); return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")
def r2(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    if y.size == 0: return float("nan")
    sst = np.sum((y - y.mean())**2) + 1e-12
    return float(1.0 - np.sum((y - yh)**2) / sst)
def brier(y, p): return float(np.mean((p - y)**2)) if len(y) else float("nan")
def nll(y, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p)))) if len(y) else float("nan")
def ece_mce(y, p, n_bins=15):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; mce = 0.0; total = len(y)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        idx = np.where((p >= lo) & (p < hi))[0]
        if idx.size == 0: continue
        conf = p[idx].mean(); acc = y[idx].mean()
        gap  = abs(acc - conf)
        ece += (idx.size/total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)
def ece_mce_adaptive(y, p, n_bins=15):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0: return float("nan"), float("nan")
    uq = np.unique(p)
    if uq.size < 3:
        return ece_mce(y, p, n_bins=min(n_bins, max(2, uq.size)))
    qs = np.quantile(p, np.linspace(0, 1, n_bins+1))
    qs[0] = 0.0; qs[-1] = 1.0
    ece = 0.0; mce = 0.0; total = len(y)
    for b in range(n_bins):
        lo, hi = qs[b], qs[b+1] + 1e-12
        idx = np.where((p >= lo) & (p < hi))[0]
        if idx.size == 0: continue
        conf = p[idx].mean(); acc = y[idx].mean()
        gap  = abs(acc - conf)
        ece += (idx.size/total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)
def roc_auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s)+1)
    sum_pos = ranks[y==1].sum()
    auc = (sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    return float(auc)
def pr_auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    n_pos = int(y.sum())
    if n_pos == 0: return float("nan")
    idx = np.argsort(-s)
    y_sorted = y[idx]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    recall = tp / max(n_pos, 1)
    precision = tp / np.maximum(tp + fp, 1)
    rec, prec = [0.0], [precision[0]]
    for r, p in zip(recall, precision):
        if r > rec[-1]:
            rec.append(float(r)); prec.append(float(p))
        else:
            prec[-1] = max(prec[-1], float(p))
    area = 0.0
    for i in range(1, len(rec)):
        area += (rec[i] - rec[i-1]) * prec[i]
    return float(area)
def thresh_metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= thr).astype(int)
    TP = int(((y==1)&(yhat==1)).sum()); TN = int(((y==0)&(yhat==0)).sum())
    FP = int(((y==0)&(yhat==1)).sum()); FN = int(((y==1)&(yhat==0)).sum())
    acc = (TP+TN)/max(len(y),1)
    prec = TP/max(TP+FP,1)
    rec  = TP/max(TP+FN,1)
    spec = TN/max(TN+FP,1)
    f1   = 2*prec*rec/max(prec+rec,1e-12)
    balc = 0.5*(rec + spec)
    return dict(ACC=acc, Precision=prec, Recall=rec, Specificity=spec, F1=f1, BalancedAcc=balc)
def youden_optimal_threshold(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y)==0: return 0.5
    grid = np.unique(np.clip(p, 0, 1))
    if grid.size > 200:
        grid = np.linspace(0, 1, 200)
    best_thr, best_j = 0.5, -1.0
    for thr in grid:
        m = thresh_metrics(y, p, thr)
        j = (2*m["BalancedAcc"] - 1.0)
        if j > best_j: best_j, best_thr = j, float(thr)
    return best_thr
def best_f1_threshold(y, p):
    ths = np.linspace(0, 1, 400)
    best = (0.5, 0.0, None)
    for th in ths:
        m = thresh_metrics(y, p, th)
        if m["F1"] > best[1]:
            best = (float(th), float(m["F1"]), m)
    return best
def threshold_for_recall(y, p, target=0.80):
    ths = np.linspace(0, 1, 400)
    best = (0.5, 1e9, None)
    for th in ths:
        m = thresh_metrics(y, p, th)
        gap = abs(m["Recall"] - target)
        if gap < best[1]:
            best = (float(th), float(gap), m)
    return best

def build_A_sequence(A0, X, drop_dim=1):
    """
    Construct time-varying adjacency A_seq[t] used from t -> t+1,
    applying the undirected deletion rule: if both endpoints are active
    on drop_dim at time t, the link is removed for all future times.
    """
    T, N, D = X.shape
    A_seq = np.zeros((T, N, N), dtype=int)
    A_curr = A0.copy()
    A_seq[0] = A_curr
    if drop_dim is not None and 0 <= int(drop_dim) < D:
        d = int(drop_dim)
        for t in range(1, T):
            active = X[t, :, d].astype(bool)
            idx = np.where(active)[0]
            if idx.size > 0:
                A_curr = A_curr.copy()
                A_curr[np.ix_(idx, idx)] = 0  # delete both directions
            A_seq[t] = A_curr
    else:
        for t in range(1, T):
            A_seq[t] = A_curr
    return A_seq


# ---- evaluation helpers (updated to pass both_d1) ----

@torch.no_grad()
def self_probs(sim, self_net):
    z = torch.tensor(sim["z"], dtype=torch.float32)
    x0 = torch.zeros_like(z)
    return self_net(z, x0).cpu().numpy()

@torch.no_grad()
def edge_attempt_probs_at_lag1(sim, edge_net, use_time=False):
    """
    Evaluate edge probabilities using the TRUE influence window from data generation.
    For mismatch experiments, attempts outside the training window get predicted prob=0.
    
    Returns q_true, q_pred arrays for all attempts in the TRUE influence window.
    """
    At, X, Y, z, phi = sim["At"], sim["X"], sim["Y"], sim["z"], sim["phi"]
    c0, c1, B = sim["true_params"]["c0"], sim["true_params"]["c1"], sim["true_params"]["B"]
    drop_dim = sim["true_params"].get("drop_dim", -1)
    m_true = sim["true_params"].get("m", 1)  # TRUE m from data generation
    m_train = sim.get("m_train", m_true)     # m used during training (may differ)
    T, N, D = X.shape
    q_true, q_pred = [], []
    
    for t in range(1, T):
        # Compute TRUE influence window (from data generation)
        window_start_true = max(0, t - m_true)
        if m_true == 1:
            recently_active_true = Y[t-1]
        else:
            recently_active_true = np.zeros((N, D), dtype=int)
            for tau in range(window_start_true, t):
                recently_active_true = np.maximum(recently_active_true, Y[tau])
        
        # Compute TRAINING influence window (what model was trained on)
        window_start_train = max(0, t - m_train)
        if m_train == 1:
            recently_active_train = Y[t-1]
        else:
            recently_active_train = np.zeros((N, D), dtype=int)
            for tau in range(window_start_train, t):
                recently_active_train = np.maximum(recently_active_train, Y[tau])
        
        A_used = At[t-1]
        if use_time:
            t_norm = torch.tensor([[t / (T - 1)]], dtype=torch.float32)
        for i in range(N):
            z_i = torch.tensor(z[i], dtype=torch.float32).unsqueeze(0)
            x_i = torch.tensor(X[t-1, i], dtype=torch.float32).unsqueeze(0)
            js = np.where(A_used[:, i] == 1)[0]
            for j in js:
                z_j = torch.tensor(z[j], dtype=torch.float32).unsqueeze(0)
                x_j = torch.tensor(X[t-1, j], dtype=torch.float32).unsqueeze(0)
                both = 0.0
                if isinstance(drop_dim, int) and (0 <= drop_dim < D):
                    both = float((X[t-1, i, drop_dim] == 1) and (X[t-1, j, drop_dim] == 1))
                both_t = torch.tensor([[both]], dtype=torch.float32)
                for k in range(D):
                    # Check if this attempt is in TRUE influence window
                    if recently_active_true[j, k] != 1: 
                        continue
                    
                    phi_ji = float(phi[j, i])
                    
                    # Model prediction: 0 if outside training window, else query model
                    if recently_active_train[j, k] == 1:
                        # Model was trained on this attempt, get prediction
                        if use_time:
                            q_vec = edge_net(
                                torch.tensor([phi_ji], dtype=torch.float32),
                                torch.tensor([k], dtype=torch.long),
                                z_i, x_i, z_j, x_j, both_t, t_norm, update_stats=False
                            ).squeeze(0).cpu().numpy()
                        else:
                            q_vec = edge_net(
                                torch.tensor([phi_ji], dtype=torch.float32),
                                torch.tensor([k], dtype=torch.long),
                                z_i, x_i, z_j, x_j, both_t, update_stats=False
                            ).squeeze(0).cpu().numpy()
                    else:
                        # Model never saw this attempt, assume no influence (all dimensions)
                        q_vec = np.zeros(D, dtype=np.float32)
                    
                    # For each target dimension d, compare true vs predicted edge prob
                    for d in range(D):
                        qt = sigmoid(c0 + c1*phi_ji + B[d, k])  # Ground truth
                        qp = float(q_vec[d])  # Model prediction (0 if outside training window)
                        q_true.append(qt)
                        q_pred.append(qp)
    return np.array(q_true), np.array(q_pred)

@torch.no_grad()
def predict_p_over_time(sim, self_net, edge_net, use_time=False):
    """
    Predict activation probabilities using the model's TRAINING window (m_train).
    This reflects the model's actual deployment behavior.
    """
    At, X, Y, z, phi = sim["At"], sim["X"], sim["Y"], sim["z"], sim["phi"]
    drop_dim = sim["true_params"].get("drop_dim", -1)
    m_train = sim.get("m_train", sim["true_params"].get("m", 1))  # Use training m
    T, N, D = X.shape
    y_all, p_all, d_all = [], [], []
    
    for t in range(1, T):
        # Use TRAINING influence window (what model was trained on)
        window_start = max(0, t - m_train)
        if m_train == 1:
            recently_active = Y[t-1]
        else:
            recently_active = np.zeros((N, D), dtype=int)
            for tau in range(window_start, t):
                recently_active = np.maximum(recently_active, Y[tau])
        
        A_used = At[t-1]
        for i in range(N):
            z_i = torch.tensor(z[i], dtype=torch.float32).unsqueeze(0)
            x_i = torch.tensor(X[t-1, i], dtype=torch.float32).unsqueeze(0)
            if use_time:
                t_norm = torch.tensor([[t / (T - 1)]], dtype=torch.float32)
                s_vec = self_net(z_i, x_i, t_norm).squeeze(0).cpu().numpy()
            else:
                s_vec = self_net(z_i, x_i).squeeze(0).cpu().numpy()
            js = np.where(A_used[:, i] == 1)[0]
            for d in range(D):
                if X[t-1, i, d] == 1: continue
                y = int(Y[t, i, d])
                s = float(s_vec[d])
                e_list = []
                for j in js:
                    z_j = torch.tensor(z[j], dtype=torch.float32).unsqueeze(0)
                    x_j = torch.tensor(X[t-1, j], dtype=torch.float32).unsqueeze(0)
                    both = 0.0
                    if isinstance(drop_dim, int) and (0 <= drop_dim < D):
                        both = float((X[t-1, i, drop_dim] == 1) and (X[t-1, j, drop_dim] == 1))
                    both_t = torch.tensor([[both]], dtype=torch.float32)
                    for k in range(D):
                        if recently_active[j, k] == 1:
                            phi_ji = float(phi[j, i])
                            if use_time:
                                q_vec = edge_net(
                                    torch.tensor([phi_ji], dtype=torch.float32),
                                    torch.tensor([k], dtype=torch.long),
                                    z_i, x_i, z_j, x_j, both_t, t_norm, update_stats=False
                                ).squeeze(0).cpu().numpy()
                            else:
                                q_vec = edge_net(
                                    torch.tensor([phi_ji], dtype=torch.float32),
                                    torch.tensor([k], dtype=torch.long),
                                    z_i, x_i, z_j, x_j, both_t, update_stats=False
                                ).squeeze(0).cpu().numpy()
                            e_list.append(float(q_vec[d]))
                prodE = float(np.prod(1.0 - np.array(e_list))) if e_list else 1.0
                p = 1.0 - (1.0 - s) * prodE
                y_all.append(y); p_all.append(p); d_all.append(d)
    return np.array(y_all), np.array(p_all), np.array(d_all)

@torch.no_grad()
def attribution_true_vs_pred(sim, self_net, edge_net, use_time=False):
    """
    Compare true vs predicted self-attribution for activated nodes.
    Uses model's TRAINING window (m_train) for prediction, but TRUE parameters for ground truth.
    """
    At, X, Y, z, phi = sim["At"], sim["X"], sim["Y"], sim["z"], sim["phi"]
    c0, c1, B = sim["true_params"]["c0"], sim["true_params"]["c1"], sim["true_params"]["B"]
    drop_dim = sim["true_params"].get("drop_dim", -1)
    m_true = sim["true_params"].get("m", 1)  # True influence window
    m_train = sim.get("m_train", m_true)     # Training influence window
    T, N, D = X.shape
    r_t, r_p = [], []
    
    for t in range(1, T):
        # Compute TRUE influence window (for ground truth)
        window_start_true = max(0, t - m_true)
        if m_true == 1:
            recently_active_true = Y[t-1]
        else:
            recently_active_true = np.zeros((N, D), dtype=int)
            for tau in range(window_start_true, t):
                recently_active_true = np.maximum(recently_active_true, Y[tau])
        
        # Compute TRAINING influence window (for model prediction)
        window_start_train = max(0, t - m_train)
        if m_train == 1:
            recently_active_train = Y[t-1]
        else:
            recently_active_train = np.zeros((N, D), dtype=int)
            for tau in range(window_start_train, t):
                recently_active_train = np.maximum(recently_active_train, Y[tau])
        
        A_used = At[t-1]
        for i in range(N):
            z_i = torch.tensor(z[i], dtype=torch.float32).unsqueeze(0)
            x_i = torch.tensor(X[t-1, i], dtype=torch.float32).unsqueeze(0)
            if use_time:
                t_norm = torch.tensor([[t / (T - 1)]], dtype=torch.float32)
                s_vec_p = self_net(z_i, x_i, t_norm).squeeze(0).cpu().numpy()
            else:
                s_vec_p = self_net(z_i, x_i).squeeze(0).cpu().numpy()
            js = np.where(A_used[:, i] == 1)[0]
            for d in range(D):
                if X[t-1, i, d] == 1 or Y[t, i, d] != 1: continue
                s_t = float(sim["s_true"][i, d]); s_p = float(s_vec_p[d])
                e_true, e_pred = [], []
                for j in js:
                    z_j = torch.tensor(z[j], dtype=torch.float32).unsqueeze(0)
                    x_j = torch.tensor(X[t-1, j], dtype=torch.float32).unsqueeze(0)
                    both = 0.0
                    if isinstance(drop_dim, int) and (0 <= drop_dim < D):
                        both = float((X[t-1, i, drop_dim] == 1) and (X[t-1, j, drop_dim] == 1))
                    both_t = torch.tensor([[both]], dtype=torch.float32)
                    for k in range(D):
                        # Ground truth uses TRUE influence window
                        if recently_active_true[j, k] == 1:
                            phi_ji = float(phi[j, i])
                            e_true.append(sigmoid(c0 + c1*phi_ji + B[d, k]))
                        
                        # Model prediction uses TRAINING influence window
                        if recently_active_train[j, k] == 1:
                            phi_ji = float(phi[j, i])
                            if use_time:
                                q_vec = edge_net(
                                    torch.tensor([phi_ji], dtype=torch.float32),
                                    torch.tensor([k], dtype=torch.long),
                                    z_i, x_i, z_j, x_j, both_t, t_norm, update_stats=False
                                ).squeeze(0).cpu().numpy()
                            else:
                                q_vec = edge_net(
                                    torch.tensor([phi_ji], dtype=torch.float32),
                                    torch.tensor([k], dtype=torch.long),
                                    z_i, x_i, z_j, x_j, both_t, update_stats=False
                                ).squeeze(0).cpu().numpy()
                            e_pred.append(float(q_vec[d]))
                prod_t = float(np.prod(1.0 - np.array(e_true))) if e_true else 1.0
                prod_p = float(np.prod(1.0 - np.array(e_pred))) if e_pred else 1.0
                p_t = 1.0 - (1.0 - s_t) * prod_t
                p_p = 1.0 - (1.0 - s_p) * prod_p
                r_t.append((s_t * prod_t) / max(p_t, 1e-12))
                r_p.append((s_p * prod_p) / max(p_p, 1e-12))
    return np.array(r_t), np.array(r_p)

# ---- plotting helpers (with correlation metrics) ----

def plot_self_scatter(sim, self_net, out, use_time=False):
    z = torch.tensor(sim["z"], dtype=torch.float32); x0 = torch.zeros_like(z)
    if use_time:
        t_norm = torch.zeros((z.shape[0], 1), dtype=torch.float32)  # Use t=0 as reference
        with torch.no_grad(): s_hat = self_net(z, x0, t_norm).cpu().numpy()
    else:
        with torch.no_grad(): s_hat = self_net(z, x0).cpu().numpy()
    s_true = sim["s_true"]
    
    # Compute correlations
    from scipy.stats import spearmanr, kendalltau
    flat_true = s_true.ravel()
    flat_pred = s_hat.ravel()
    spearman_r, spearman_p = spearmanr(flat_true, flat_pred)
    kendall_tau, kendall_p = kendalltau(flat_true, flat_pred)
    
    plt.figure(figsize=(6,5))
    plt.scatter(flat_true, flat_pred, s=14, alpha=0.6)
    lim = [0, max(0.01, flat_true.max(), flat_pred.max())]
    plt.plot(lim, lim, 'k--', lw=1, label='Perfect fit')
    plt.xlabel("True self prob"); plt.ylabel("Estimated self prob (x_prev=0)")
    plt.title("Self activation (all dims)")
    
    # Add correlation text
    textstr = f'Spearman ρ = {spearman_r:.4f}\nKendall τ = {kendall_tau:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

def plot_edge_scatter(sim, edge_net, out, use_time=False):
    qt, qp = edge_attempt_probs_at_lag1(sim, edge_net, use_time=use_time)
    if qt.size==0: return
    
    # Compute correlations
    from scipy.stats import spearmanr, kendalltau
    spearman_r, spearman_p = spearmanr(qt, qp)
    kendall_tau, kendall_p = kendalltau(qt, qp)
    
    plt.figure(figsize=(6,5))
    plt.scatter(qt, qp, s=10, alpha=0.6)
    lim = [0, max(0.01, qt.max(), qp.max())]
    plt.plot(lim, lim, 'k--', lw=1, label='Perfect fit')
    plt.xlabel("True per-attempt edge prob"); plt.ylabel("Estimated")
    plt.title("Edge mechanism on actual attempts (all dest dims)")
    
    # Add correlation text
    textstr = f'Spearman ρ = {spearman_r:.4f}\nKendall τ = {kendall_tau:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

def plot_simulation_dynamics(sim, outdir):
    """Plot additional simulation dynamics: incremental activation rate."""
    Y = sim["Y"]  # (T, N, D)
    T, N, D = Y.shape
    
    # Incremental activation rate per dimension
    plt.figure(figsize=(8, 5))
    for d in range(D):
        increments = Y[:, :, d].sum(axis=1)  # (T,)
        plt.plot(np.arange(T), increments, linewidth=2, label=f'Dimension {d}')
    plt.xlabel("Time step", fontsize=12)
    plt.ylabel("New activations per step", fontsize=12)
    plt.title("Simulation: Incremental Activation Rate", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sim_incremental_activation.png"), dpi=140)
    plt.close()

# ---- CLI / main ----

def parse_args():
    ap = argparse.ArgumentParser("Visualize + evaluate Neural IC + EM (joint, dynamic At with link drop on a chosen dimension)")
    # simulation
    ap.add_argument("--N", type=int, default=60)
    ap.add_argument("--T", type=int, default=150)
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--p_edge", type=float, default=0.10)
    ap.add_argument("--p_seed", type=float, default=0.02)
    ap.add_argument("--rho", type=float, default=0.4)
    ap.add_argument("--drop_dim", type=int, default=1, help="dimension index whose co-activation deletes edges")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--c0", type=float, default=-4.0)
    ap.add_argument("--c1", type=float, default=2.0)
    ap.add_argument("--a0", type=str, default="-3.0,-3.3,-2.7", help="comma-separated self baseline per dim")
    # influence window
    ap.add_argument("--m", type=int, default=1, help="max influence window: nodes active in past m steps can influence (default=1, classic IC)")
    ap.add_argument("--m_train", type=int, default=None, help="override m for training (for mismatch experiments); if None, use --m")
    # training
    ap.add_argument("--self_hidden", type=int, default=64)
    ap.add_argument("--self_depth", type=int, default=2)
    ap.add_argument("--edge_emb", type=int, default=8)
    ap.add_argument("--edge_hidden", type=int, default=96)
    ap.add_argument("--edge_depth", type=int, default=3)
    ap.add_argument("--em_iters", type=int, default=24)
    ap.add_argument("--epochs_self", type=int, default=10)
    ap.add_argument("--epochs_edge", type=int, default=8)
    ap.add_argument("--lr_self", type=float, default=3e-3)
    ap.add_argument("--lr_edge", type=float, default=8e-3)
    ap.add_argument("--wd_self", type=float, default=1e-4)
    ap.add_argument("--wd_edge", type=float, default=3e-3)
    ap.add_argument("--phi_monotone", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu")
    # EM stabilizers (fixed)
    ap.add_argument("--tau_resp", type=float, default=0.85)
    ap.add_argument("--edge_resp_floor", type=float, default=0.12)
    # Annealing (optional; if provided, overrides fixed per-iter)
    ap.add_argument("--tau_start", type=float, default=None)
    ap.add_argument("--tau_end", type=float, default=None)
    ap.add_argument("--edge_floor_start", type=float, default=None)
    ap.add_argument("--edge_floor_end", type=float, default=None)
    # φ warmup
    ap.add_argument("--phi_warmup", type=int, default=0, help="iters to refresh phi mean/var from attempts")
    # ranking regularizer
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--rank_pairs", type=int, default=1024)
    ap.add_argument("--rank_margin", type=float, default=0.02)
    # per-state self weighting
    ap.add_argument("--self_weighting", type=str, default="sqrt_invfreq",
                    choices=["none", "invfreq", "sqrt_invfreq"])
    # evaluation
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--target_recall", type=float, default=0.80)
    # output
    ap.add_argument("--outdir", type=str, default="viz_results_dynamic")
    ap.add_argument("--make_plots", type=int, default=1)
    ap.add_argument("--K_self", type=int, default=0, help="suppress self activations before this step (0 = off)")
    ap.add_argument("--K_edge", type=int, default=0, help="suppress edge activations before this step (0 = off)")
    ap.add_argument("--bias_self", type=float, default=10.0, help="logit suppression magnitude for self")
    ap.add_argument("--bias_edge", type=float, default=10.0, help="logit suppression magnitude for edge")
    ap.add_argument("--mode_self", type=str, default="hard", choices=["hard","linear"])
    ap.add_argument("--mode_edge", type=str, default="hard", choices=["hard","linear"])
    # Time modulation mode
    ap.add_argument("--time_mode", type=str, default="warmup", choices=["warmup", "inverse_u"],
                    help="Time modulation: warmup (legacy) or inverse_u (new)")
    ap.add_argument("--peak_times", type=str, default=None,
                    help="comma-separated peak times per dim for inverse_u mode")
    ap.add_argument("--widths", type=str, default=None,
                    help="comma-separated widths per dim for inverse_u mode")
    ap.add_argument("--baseline", type=float, default=0.05,
                    help="minimum multiplier for inverse_u mode")
    ap.add_argument("--max_suppression", type=float, default=5.0,
                    help="maximum suppression strength (logit units) for inverse_u mode")
    
    # Time feature
    ap.add_argument("--use_time", action="store_true",
                    help="whether to include normalized time as input feature to models")
    
    # Train/test split
    ap.add_argument("--T_train", type=int, default=None,
                    help="Train/test split: train on [0, T_train), rollout on [T_train, T_test). "
                         "If None, use all data (no split).")
    ap.add_argument("--T_test", type=int, default=50,
                    help="End of test evaluation window (exclusive). Default: 50")
    ap.add_argument("--rollout_seed", type=int, default=None,
                    help="RNG seed for rollout (if None, use same as --seed)")

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Parse a0 parameter
    a0_vals = tuple(float(x) for x in args.a0.split(','))
    
    # Parse time mode parameters
    peak_times_vals = None
    widths_vals = None
    if args.time_mode == "inverse_u":
        if args.peak_times:
            peak_times_vals = [float(x) for x in args.peak_times.split(',')]
        if args.widths:
            widths_vals = [float(x) for x in args.widths.split(',')]
    
    # Run simulation
    sim = simulate_icD_dynamic(
        N=args.N, T=args.T, D=args.D, p_edge=args.p_edge, p_seed=args.p_seed,
        c0=args.c0, c1=args.c1, a0=a0_vals,
        seed=args.seed, rho=args.rho, drop_dim=args.drop_dim,
        m=args.m,  # influence window parameter
        time_mode=args.time_mode,
        K_self=args.K_self, K_edge=args.K_edge,
        bias_self=args.bias_self, bias_edge=args.bias_edge,
        mode_self=args.mode_self, mode_edge=args.mode_edge,
        peak_times=peak_times_vals, widths=widths_vals, baseline=args.baseline,
        max_suppression=args.max_suppression
    )

    # Build and attach time-varying adjacency
    sim["At"] = build_A_sequence(sim["A"], sim["X"], drop_dim=args.drop_dim)

    # === Train/test split logic ===
    if args.T_train is not None and args.T_train > 0:
        print(f"\n{'='*60}")
        print(f"TRAIN/TEST SPLIT MODE")
        print(f"Train period: [0, {args.T_train})")
        print(f"Test period:  [{args.T_train}, {args.T_test})")
        print(f"{'='*60}\n")
        
        # Train on partial data
        self_net, edge_net = train_em(
            sim,
            D=args.D,
            self_hidden=args.self_hidden, self_depth=args.self_depth,
            edge_emb=args.edge_emb, edge_hidden=args.edge_hidden, edge_depth=args.edge_depth,
            phi_monotone=bool(args.phi_monotone),
            em_iters=args.em_iters, epochs_self=args.epochs_self, epochs_edge=args.epochs_edge,
            lr_self=args.lr_self, lr_edge=args.lr_edge, wd_self=args.wd_self, wd_edge=args.wd_edge,
            device=args.device, verbose=True,
            m_train=args.m_train,
            T_max=args.T_train,  # Only use data up to T_train
            tau_resp=args.tau_resp, edge_resp_floor=args.edge_resp_floor,
            tau_start=args.tau_start, tau_end=args.tau_end,
            edge_floor_start=args.edge_floor_start, edge_floor_end=args.edge_floor_end,
            phi_warmup_iters=args.phi_warmup,
            freeze_phi_stats=True,
            lambda_rank=args.lambda_rank, rank_pairs=args.rank_pairs, rank_margin=args.rank_margin,
            self_weighting=args.self_weighting,
            use_time=args.use_time
        )
        
        # Rollout prediction
        from rollout_predictor import rollout_from_trained, compute_test_cascade_metrics, plot_cumulative_activation_split
        
        rollout_seed = args.rollout_seed if args.rollout_seed is not None else args.seed
        print(f"\n[Rollout] Starting rollout from t={args.T_train} to t={args.T_test} (seed={rollout_seed})...")
        
        sim_rollout = rollout_from_trained(
            sim, self_net, edge_net, args.T_train,
            device=args.device, use_time=args.use_time, seed=rollout_seed
        )
        
        print(f"[Rollout] Completed. Computing test metrics...")
        
        # Compute train metrics (on [0, T_train))
        # Create a sliced sim for train period
        sim_train = {
            "X": sim["X"][:args.T_train],
            "Y": sim["Y"][:args.T_train],
            "At": sim["At"][:args.T_train],
            "A": sim["A"],
            "z": sim["z"],
            "phi": sim["phi"],
            "s_true": sim["s_true"],
            "true_params": sim["true_params"],
        }
        if "m_train" in sim:
            sim_train["m_train"] = sim["m_train"]
        
        # Train metrics (all existing metrics on train period)
        print("\n=== Train Metrics (on ground truth) ===")
        
        def self_probs_eval(sim_data, self_net_model):
            z = torch.tensor(sim_data["z"], dtype=torch.float32)
            x0 = torch.zeros_like(z)
            if args.use_time:
                t_norm = torch.zeros((z.shape[0], 1), dtype=torch.float32)
                with torch.no_grad():
                    return self_net_model(z, x0, t_norm).cpu().numpy()
            else:
                with torch.no_grad():
                    return self_net_model(z, x0).cpu().numpy()
        
        s_hat = self_probs_eval(sim_train, self_net)
        Self_RMSE = rmse(sim_train["s_true"], s_hat)
        Self_R2 = r2(sim_train["s_true"].ravel(), s_hat.ravel())
        
        train_metrics = {
            "Self_RMSE": Self_RMSE,
            "Self_R2": Self_R2,
        }
        
        for d in range(sim_train["s_true"].shape[1]):
            train_metrics[f"Self_RMSE_d{d}"] = rmse(sim_train["s_true"][:,d], s_hat[:,d])
            train_metrics[f"Self_R2_d{d}"] = r2(sim_train["s_true"][:,d], s_hat[:,d])
        
        qt, qp = edge_attempt_probs_at_lag1(sim_train, edge_net, use_time=args.use_time)
        train_metrics["B_RMSE"] = rmse(qt, qp) if qt.size else float("nan")
        train_metrics["B_R2"] = r2(qt, qp) if qt.size else float("nan")
        
        r_t, r_p = attribution_true_vs_pred(sim_train, self_net, edge_net, use_time=args.use_time)
        train_metrics["Attr_RMSE_vsTruePosterior"] = rmse(r_t, r_p) if r_t.size else float("nan")
        train_metrics["Attr_R2_vsTruePosterior"] = r2(r_t, r_p) if r_t.size else float("nan")
        
        y_all, p_all, d_all = predict_p_over_time(sim_train, self_net, edge_net, use_time=args.use_time)
        train_metrics["NLL"] = nll(y_all, p_all)
        train_metrics["Brier"] = brier(y_all, p_all)
        
        ECE, MCE = ece_mce(y_all, p_all, n_bins=args.n_bins)
        ECEa, MCEa = ece_mce_adaptive(y_all, p_all, n_bins=args.n_bins)
        train_metrics["ECE"] = ECE
        train_metrics["MCE"] = MCE
        train_metrics["ECE_adaptive"] = ECEa
        train_metrics["MCE_adaptive"] = MCEa
        
        train_metrics["ROC_AUC"] = roc_auc(y_all, p_all)
        train_metrics["PR_AUC"] = pr_auc(y_all, p_all)
        
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
        # Test metrics (cascade-level on [T_train, T_test))
        print("\n=== Test Metrics (cascade-level, rollout vs ground truth) ===")
        test_metrics = compute_test_cascade_metrics(sim, sim_rollout, args.T_train, args.T_test)
        
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
        # Save metrics
        all_metrics = {
            "train": train_metrics,
            "test": test_metrics,
            "config": {
                "T_train": args.T_train,
                "T_test": args.T_test,
                "rollout_seed": rollout_seed,
                "drop_dim": args.drop_dim,
                "m": args.m,
                "m_train": args.m_train,
                "time_mode": args.time_mode,
            }
        }
        
        with open(os.path.join(args.outdir, "metrics_split.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n[Split mode] Saved metrics to {args.outdir}/metrics_split.json")
        
        # Visualization
        if args.make_plots:
            plot_cumulative_activation_split(sim, sim_rollout, args.T_train, args.T_test, args.outdir)
            
            # Also save train-period plots
            plot_self_scatter(sim_train, self_net, os.path.join(args.outdir, "train_self_scatter.png"), use_time=args.use_time)
            plot_edge_scatter(sim_train, edge_net, os.path.join(args.outdir, "train_edge_scatter.png"), use_time=args.use_time)
        
        return  # Exit after split mode
    
    # === Full data mode (no split) ===
    print(f"\n{'='*60}")
    print(f"FULL DATA MODE (no train/test split)")
    print(f"{'='*60}\n")
    
    self_net, edge_net = train_em(
        sim,
        D=args.D,
        self_hidden=args.self_hidden, self_depth=args.self_depth,
        edge_emb=args.edge_emb, edge_hidden=args.edge_hidden, edge_depth=args.edge_depth, phi_monotone=bool(args.phi_monotone),
        em_iters=args.em_iters, epochs_self=args.epochs_self, epochs_edge=args.epochs_edge,
        lr_self=args.lr_self, lr_edge=args.lr_edge, wd_self=args.wd_self, wd_edge=args.wd_edge,
        device=args.device, verbose=True,
        m_train=args.m_train,  # for mismatch experiments
        tau_resp=args.tau_resp, edge_resp_floor=args.edge_resp_floor,
        # annealing knobs (optional)
        tau_start=args.tau_start, tau_end=args.tau_end,
        edge_floor_start=args.edge_floor_start, edge_floor_end=args.edge_floor_end,
        # φ warmup
        phi_warmup_iters=args.phi_warmup,
        freeze_phi_stats=True,
        lambda_rank=args.lambda_rank, rank_pairs=args.rank_pairs, rank_margin=args.rank_margin,
        self_weighting=args.self_weighting,
        use_time=args.use_time
    )

    # --- metrics
    def self_probs(sim, self_net):
        z = torch.tensor(sim["z"], dtype=torch.float32); x0 = torch.zeros_like(z)
        # For evaluation, use t_norm=0 (beginning) as a reference point when use_time=True
        if args.use_time:
            t_norm = torch.zeros((z.shape[0], 1), dtype=torch.float32)
            with torch.no_grad(): return self_net(z, x0, t_norm).cpu().numpy()
        else:
            with torch.no_grad(): return self_net(z, x0).cpu().numpy()
    s_hat = self_probs(sim, self_net)
    Self_RMSE = rmse(sim["s_true"], s_hat); Self_R2 = r2(sim["s_true"].ravel(), s_hat.ravel())
    per_state = {}
    for d in range(sim["s_true"].shape[1]):
        per_state[f"Self_RMSE_d{d}"] = rmse(sim["s_true"][:,d], s_hat[:,d])
        per_state[f"Self_R2_d{d}"]   = r2(sim["s_true"][:,d], s_hat[:,d])

    qt, qp = edge_attempt_probs_at_lag1(sim, edge_net, use_time=args.use_time)
    B_RMSE = rmse(qt, qp) if qt.size else float("nan")
    B_R2   = r2(qt, qp) if qt.size else float("nan")

    r_t, r_p = attribution_true_vs_pred(sim, self_net, edge_net, use_time=args.use_time)
    Attr_RMSE = rmse(r_t, r_p) if r_t.size else float("nan")
    Attr_R2   = r2(r_t, r_p) if r_t.size else float("nan")

    y_all, p_all, d_all = predict_p_over_time(sim, self_net, edge_net, use_time=args.use_time)
    NLL   = nll(y_all, p_all); BrierS= brier(y_all, p_all)
    ECE, MCE = ece_mce(y_all, p_all, n_bins=args.n_bins)
    ECEa, MCEa = ece_mce_adaptive(y_all, p_all, n_bins=args.n_bins)
    AUC   = roc_auc(y_all, p_all); AUPRC = pr_auc(y_all, p_all)
    thr50 = 0.5; m50 = thresh_metrics(y_all, p_all, thr50)
    thrY  = youden_optimal_threshold(y_all, p_all); mY = thresh_metrics(y_all, p_all, thrY)
    thrF1, F1star, mF1 = best_f1_threshold(y_all, p_all)
    thrR, gapR, mR = threshold_for_recall(y_all, p_all, target=args.target_recall)

    metrics = dict(
        Self_RMSE=Self_RMSE, Self_R2=Self_R2,
        B_RMSE=B_RMSE, B_R2=B_R2,
        Attr_RMSE_vsTruePosterior=Attr_RMSE, Attr_R2_vsTruePosterior=Attr_R2,
        NLL=NLL, Brier=BrierS, ECE=ECE, MCE=MCE, ECE_adaptive=ECEa, MCE_adaptive=MCEa,
        ROC_AUC=AUC, PR_AUC=AUPRC,
        ACC_t0p5=m50["ACC"], F1_t0p5=m50["F1"], Precision_t0p5=m50["Precision"],
        Recall_t0p5=m50["Recall"], Specificity_t0p5=m50["Specificity"], BalancedAcc_t0p5=m50["BalancedAcc"],
        Thr_Youden=thrY,
        ACC_tY=mY["ACC"], F1_tY=mY["F1"], Precision_tY=mY["Precision"],
        Recall_tY=mY["Recall"], Specificity_tY=mY["Specificity"], BalancedAcc_tY=mY["BalancedAcc"],
        Thr_bestF1=thrF1, F1_bestF1=F1star,
        ACC_tF1=mF1["ACC"], Precision_tF1=mF1["Precision"], Recall_tF1=mF1["Recall"],
        Specificity_tF1=mF1["Specificity"], BalancedAcc_tF1=mF1["BalancedAcc"],
        Thr_recallTarget=thrR, RecallTarget=args.target_recall,
        ACC_tR=mR["ACC"], F1_tR=mR["F1"], Precision_tR=mR["Precision"],
        Recall_tR=mR["Recall"], Specificity_tR=mR["Specificity"], BalancedAcc_tR=mR["BalancedAcc"],
        drop_dim=args.drop_dim, self_weighting=args.self_weighting,
        tau_resp=args.tau_resp, edge_resp_floor=args.edge_resp_floor, lambda_rank=args.lambda_rank
    )
    metrics.update(per_state)

    print("\n=== Full Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.make_plots:
        plot_self_scatter(sim, self_net, os.path.join(args.outdir, "self_scatter.png"), use_time=args.use_time)
        plot_edge_scatter(sim, edge_net, os.path.join(args.outdir, "edge_scatter.png"), use_time=args.use_time)
        plot_simulation_dynamics(sim, args.outdir)
        
        # Also generate network dynamics visualizations
        from visualize_dynamic_network import visualize_network_dynamics
        viz_paths = visualize_network_dynamics(
            sim, outdir=args.outdir, drop_dim=args.drop_dim,
            step=1, fps=8, three_panel=True, per_dim=True
        )
        print(f"\nSaved figures to: {args.outdir}")

if __name__ == "__main__":
    main()
