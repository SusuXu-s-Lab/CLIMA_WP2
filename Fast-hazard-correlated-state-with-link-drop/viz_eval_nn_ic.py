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
    Logit bias for slow starts:
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

def simulate_icD_dynamic(
    N=60, T=150, D=3, p_edge=0.10, p_seed=0.02,
    c0=-4.0, c1=2.0, seed=7, rho=0.0, drop_dim=1,
    a0=(-3.0, -3.3, -2.7), a1=(1.2, 0.9, 1.5), B=None,
    # NEW (optional, defaults keep old behavior):
    K_self=0,           # no self gating by default
    K_edge=0,           # no edge gating by default
    bias_self=10.0,     # strength of early suppression (logit units)
    bias_edge=10.0,
    mode_self="hard",   # 'hard' or 'linear'
    mode_edge="hard"
):
    """
    IC simulator with optional time-gated hazards, preserving the original API.
    If K_self or K_edge > 0, activations are strongly suppressed before those steps.
    Returns fields expected by the trainer/evaluator: A, z, phi, X, Y, s_true, true_params.
    """
    rng = np.random.default_rng(seed)

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

    a0 = np.array(a0)[:D]
    a1 = np.array(a1)[:D]
    if B is None:
        B = np.array([[ 0.5, -0.2,  0.1],
                      [ 0.3,  0.6, -0.1],
                      [-0.1,  0.4,  0.8]], float)[:D, :D]
    else:
        B = np.array(B, float)[:D, :D]

    s_base = sigmoid(a0[None, :] + a1[None, :] * z)   # (N,D)
    s_base_logit = logit(s_base)

    X = np.zeros((T, N, D), dtype=int)  # cumulative
    Y = np.zeros((T, N, D), dtype=int)  # increments
    X[0] = (rng.random((N, D)) < p_seed).astype(int)

    # We evolve A dynamically (drop on drop_dim), but return A0 to the trainer
    # so attempts are enumerated consistently; Y carries the actual dynamics.
    A0 = A.copy()

    for t in range(1, T):
        newly = Y[t - 1]
        b_s = time_bias(t, K_self, big=bias_self, mode=mode_self)
        b_e = time_bias(t, K_edge, big=bias_edge, mode=mode_edge)

        for i in range(N):
            for d in range(D):
                if X[t - 1, i, d] == 1:
                    continue
                # time-gated self probability
                s_t = sigmoid(s_base_logit[i, d] + b_s)

                # time-gated edge attempts from neighbors newly active at t-1
                net_hit = False
                if np.any(newly):
                    js = np.where(A[:, i] == 1)[0]
                    for j in js:
                        for k in range(D):
                            if newly[j, k] == 1:
                                q_logit = c0 + c1 * phi[j, i] + B[d, k] + b_e
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
        true_params=dict(c0=c0, c1=c1, B=B,
                         K_self=K_self, K_edge=K_edge,
                         bias_self=bias_self, bias_edge=bias_edge,
                         mode_self=mode_self, mode_edge=mode_edge)
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
def edge_attempt_probs_at_lag1(sim, edge_net):
    At, X, Y, z, phi = sim["At"], sim["X"], sim["Y"], sim["z"], sim["phi"]
    c0, c1, B = sim["true_params"]["c0"], sim["true_params"]["c1"], sim["true_params"]["B"]
    drop_dim = sim["true_params"].get("drop_dim", -1)
    T, N, D = X.shape
    q_true, q_pred = [], []
    for t in range(1, T):
        newly = Y[t-1]
        A_used = At[t-1]
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
                    if newly[j, k] != 1: continue
                    phi_ji = float(phi[j, i])
                    q_vec = edge_net(
                        torch.tensor([phi_ji], dtype=torch.float32),
                        torch.tensor([k], dtype=torch.long),
                        z_i, x_i, z_j, x_j, both_t, update_stats=False
                    ).squeeze(0).cpu().numpy()
                    for d in range(D):
                        qt = sigmoid(c0 + c1*phi_ji + B[d, k])
                        q_true.append(qt); q_pred.append(float(q_vec[d]))
    return np.array(q_true), np.array(q_pred)

@torch.no_grad()
def predict_p_over_time(sim, self_net, edge_net):
    At, X, Y, z, phi = sim["At"], sim["X"], sim["Y"], sim["z"], sim["phi"]
    drop_dim = sim["true_params"].get("drop_dim", -1)
    T, N, D = X.shape
    y_all, p_all, d_all = [], [], []
    for t in range(1, T):
        newly = Y[t-1]
        A_used = At[t-1]
        for i in range(N):
            z_i = torch.tensor(z[i], dtype=torch.float32).unsqueeze(0)
            x_i = torch.tensor(X[t-1, i], dtype=torch.float32).unsqueeze(0)
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
                        if newly[j, k] == 1:
                            phi_ji = float(phi[j, i])
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
def attribution_true_vs_pred(sim, self_net, edge_net):
    At, X, Y, z, phi = sim["At"], sim["X"], sim["Y"], sim["z"], sim["phi"]
    c0, c1, B = sim["true_params"]["c0"], sim["true_params"]["c1"], sim["true_params"]["B"]
    drop_dim = sim["true_params"].get("drop_dim", -1)
    T, N, D = X.shape
    r_t, r_p = [], []
    for t in range(1, T):
        newly = Y[t-1]
        A_used = At[t-1]
        for i in range(N):
            z_i = torch.tensor(z[i], dtype=torch.float32).unsqueeze(0)
            x_i = torch.tensor(X[t-1, i], dtype=torch.float32).unsqueeze(0)
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
                        if newly[j, k] == 1:
                            phi_ji = float(phi[j, i])
                            e_true.append(sigmoid(c0 + c1*phi_ji + B[d, k]))
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

# ---- plotting helpers (unchanged essentials) ----

def plot_self_scatter(sim, self_net, out):
    z = torch.tensor(sim["z"], dtype=torch.float32); x0 = torch.zeros_like(z)
    with torch.no_grad(): s_hat = self_net(z, x0).cpu().numpy()
    s_true = sim["s_true"]
    plt.figure(figsize=(5,5))
    plt.scatter(s_true.ravel(), s_hat.ravel(), s=14, alpha=0.6)
    lim = [0, max(0.01, s_true.max(), s_hat.max())]
    plt.plot(lim, lim, 'k--', lw=1)
    plt.xlabel("True self prob"); plt.ylabel("Estimated self prob (x_prev=0)")
    plt.title("Self activation (all dims)")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

def plot_edge_scatter(sim, edge_net, out):
    qt, qp = edge_attempt_probs_at_lag1(sim, edge_net)
    if qt.size==0: return
    plt.figure(figsize=(5,5))
    plt.scatter(qt, qp, s=10, alpha=0.6)
    lim = [0, max(0.01, qt.max(), qp.max())]
    plt.plot(lim, lim, 'k--', lw=1)
    plt.xlabel("True per-attempt edge prob"); plt.ylabel("Estimated")
    plt.title("Edge mechanism on actual attempts (all dest dims)")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

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

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # sim = simulate_icD_dynamic(
    #     N=args.N, T=args.T, D=args.D, p_edge=args.p_edge, p_seed=args.p_seed,
    #     c0=args.c0, c1=args.c1, seed=args.seed, rho=args.rho, drop_dim=args.drop_dim
    # )
    sim = simulate_icD_dynamic(
    N=args.N, T=args.T, D=args.D, p_edge=args.p_edge, p_seed=args.p_seed,
    c0=args.c0, c1=args.c1, seed=args.seed, rho=args.rho, drop_dim=args.drop_dim,
    K_self=args.K_self, K_edge=args.K_edge,
    bias_self=args.bias_self, bias_edge=args.bias_edge,
    mode_self=args.mode_self, mode_edge=args.mode_edge)

    # build and attach time-varying adjacency for downstream code that expects sim["At"]
    sim["At"] = build_A_sequence(sim["A"], sim["X"], drop_dim=args.drop_dim)

    from visualize_dynamic_network import visualize_network_dynamics
    paths = visualize_network_dynamics(sim, outdir="sim_viz", drop_dim=args.drop_dim, step=1, fps=2)


    self_net, edge_net = train_em(
        sim,
        D=args.D,
        self_hidden=args.self_hidden, self_depth=args.self_depth,
        edge_emb=args.edge_emb, edge_hidden=args.edge_hidden, edge_depth=args.edge_depth, phi_monotone=bool(args.phi_monotone),
        em_iters=args.em_iters, epochs_self=args.epochs_self, epochs_edge=args.epochs_edge,
        lr_self=args.lr_self, lr_edge=args.lr_edge, wd_self=args.wd_self, wd_edge=args.wd_edge,
        device=args.device, verbose=True,
        tau_resp=args.tau_resp, edge_resp_floor=args.edge_resp_floor,
        # annealing knobs (optional)
        tau_start=args.tau_start, tau_end=args.tau_end,
        edge_floor_start=args.edge_floor_start, edge_floor_end=args.edge_floor_end,
        # φ warmup
        phi_warmup_iters=args.phi_warmup,
        freeze_phi_stats=True,
        lambda_rank=args.lambda_rank, rank_pairs=args.rank_pairs, rank_margin=args.rank_margin,
        self_weighting=args.self_weighting
    )

    # --- metrics
    def self_probs(sim, self_net):
        z = torch.tensor(sim["z"], dtype=torch.float32); x0 = torch.zeros_like(z)
        with torch.no_grad(): return self_net(z, x0).cpu().numpy()
    s_hat = self_probs(sim, self_net)
    Self_RMSE = rmse(sim["s_true"], s_hat); Self_R2 = r2(sim["s_true"].ravel(), s_hat.ravel())
    per_state = {}
    for d in range(sim["s_true"].shape[1]):
        per_state[f"Self_RMSE_d{d}"] = rmse(sim["s_true"][:,d], s_hat[:,d])
        per_state[f"Self_R2_d{d}"]   = r2(sim["s_true"][:,d], s_hat[:,d])

    qt, qp = edge_attempt_probs_at_lag1(sim, edge_net)
    B_RMSE = rmse(qt, qp) if qt.size else float("nan")
    B_R2   = r2(qt, qp) if qt.size else float("nan")

    r_t, r_p = attribution_true_vs_pred(sim, self_net, edge_net)
    Attr_RMSE = rmse(r_t, r_p) if r_t.size else float("nan")
    Attr_R2   = r2(r_t, r_p) if r_t.size else float("nan")

    y_all, p_all, d_all = predict_p_over_time(sim, self_net, edge_net)
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
        plot_self_scatter(sim, self_net, os.path.join(args.outdir, "self_scatter.png"))
        plot_edge_scatter(sim, edge_net, os.path.join(args.outdir, "edge_scatter.png"))
        print(f"\nSaved figures to: {args.outdir}")

if __name__ == "__main__":
    main()
