# rollout_predictor.py
# Rollout prediction and test evaluation for train/test split experiments.

import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def rollout_from_trained(sim, self_net, edge_net, T_train, device="cpu", use_time=False, seed=None):
    """
    Rollout cascade prediction from T_train to T using trained models.
    Sampling logic matches simulate_icD_dynamic exactly (per-attempt sampling).
    
    Args:
        sim: full simulation dict (ground truth)
        self_net, edge_net: trained models
        T_train: train on [0, T_train), rollout on [T_train, T)
        device: torch device
        use_time: whether models use time feature
        seed: RNG seed for reproducible rollout
    
    Returns:
        sim_rollout: dict with same structure as sim
                     X, Y, At are rollout results (first T_train steps copied from GT)
    """
    rng = np.random.default_rng(seed)
    
    # Extract simulation parameters
    T, N, D = sim["X"].shape
    drop_dim = sim["true_params"].get("drop_dim", -1)
    m_train = sim.get("m_train", sim["true_params"].get("m", 1))
    
    # Initialize rollout arrays (copy train period from ground truth)
    # Include T_train to ensure continuity at the split point
    X_rollout = np.zeros((T, N, D), dtype=int)
    Y_rollout = np.zeros((T, N, D), dtype=int)
    X_rollout[:T_train+1] = sim["X"][:T_train+1].copy()
    Y_rollout[:T_train+1] = sim["Y"][:T_train+1].copy()
    
    # Build dynamic adjacency based on rollout states
    A_rollout = np.zeros((T, N, N), dtype=int)
    A_curr = sim["A"].copy()
    
    # Reconstruct adjacency for train period using rollout X (same as GT in this period)
    # Include T_train to match the X/Y copying above
    for t in range(T_train + 1):
        if t > 0 and drop_dim is not None and 0 <= drop_dim < D:
            active = X_rollout[t, :, drop_dim].astype(bool)
            idx = np.where(active)[0]
            if idx.size > 0:
                A_curr = A_curr.copy()
                A_curr[np.ix_(idx, idx)] = 0
        A_rollout[t] = A_curr.copy()
    
    # Rollout from T_train+1 to T (start prediction AFTER the split point)
    self_net.eval()
    edge_net.eval()
    
    for t in range(T_train + 1, T):
        # Compute influence window based on rollout history
        window_start = max(0, t - m_train)
        if m_train == 1:
            recently_active = Y_rollout[t-1]
        else:
            recently_active = np.zeros((N, D), dtype=int)
            for tau in range(window_start, t):
                recently_active = np.maximum(recently_active, Y_rollout[tau])
        
        # For each node and dimension
        for i in range(N):
            for d in range(D):
                if X_rollout[t-1, i, d] == 1:
                    continue  # Already activated
                
                # Self activation probability
                z_i = torch.tensor(sim["z"][i], dtype=torch.float32, device=device).unsqueeze(0)
                x_i = torch.tensor(X_rollout[t-1, i], dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    if use_time:
                        t_norm = torch.tensor([[t / (T - 1)]], dtype=torch.float32, device=device)
                        s_vec = self_net(z_i, x_i, t_norm).squeeze(0).cpu().numpy()
                    else:
                        s_vec = self_net(z_i, x_i).squeeze(0).cpu().numpy()
                
                s_d = float(s_vec[d])
                
                # Sample self activation (same as simulator)
                self_hit = (rng.random() < s_d)
                
                # Edge attempts (only if self didn't activate)
                net_hit = False
                if not self_hit:
                    js = np.where(A_curr[:, i] == 1)[0]
                    for j in js:
                        z_j = torch.tensor(sim["z"][j], dtype=torch.float32, device=device).unsqueeze(0)
                        x_j = torch.tensor(X_rollout[t-1, j], dtype=torch.float32, device=device).unsqueeze(0)
                        
                        # Compute both_d1 flag
                        both = 0.0
                        if 0 <= drop_dim < D:
                            both = float((X_rollout[t-1, i, drop_dim] == 1) and 
                                       (X_rollout[t-1, j, drop_dim] == 1))
                        both_t = torch.tensor([[both]], dtype=torch.float32, device=device)
                        
                        for k in range(D):
                            if recently_active[j, k] == 1:
                                phi_ji = float(sim["phi"][j, i])
                                
                                with torch.no_grad():
                                    if use_time:
                                        t_norm = torch.tensor([[t / (T - 1)]], dtype=torch.float32, device=device)
                                        q_vec = edge_net(
                                            torch.tensor([phi_ji], dtype=torch.float32, device=device),
                                            torch.tensor([k], dtype=torch.long, device=device),
                                            z_i, x_i, z_j, x_j, both_t, t_norm, update_stats=False
                                        ).squeeze(0).cpu().numpy()
                                    else:
                                        q_vec = edge_net(
                                            torch.tensor([phi_ji], dtype=torch.float32, device=device),
                                            torch.tensor([k], dtype=torch.long, device=device),
                                            z_i, x_i, z_j, x_j, both_t, update_stats=False
                                        ).squeeze(0).cpu().numpy()
                                
                                q_d = float(q_vec[d])
                                
                                # Sample THIS attempt (same as simulator!)
                                if rng.random() < q_d:
                                    net_hit = True
                                    break
                        
                        if net_hit:
                            break
                
                # Record activation
                Y_rollout[t, i, d] = int(self_hit or net_hit)
        
        # Update cumulative state
        X_rollout[t] = np.maximum(X_rollout[t-1], Y_rollout[t])
        
        # Update adjacency for next step
        if t < T - 1:
            if drop_dim is not None and 0 <= drop_dim < D:
                active = X_rollout[t, :, drop_dim].astype(bool)
                idx = np.where(active)[0]
                if idx.size > 0:
                    A_curr = A_curr.copy()
                    A_curr[np.ix_(idx, idx)] = 0
            A_rollout[t] = A_curr.copy()
    
    # Build rollout simulation dict
    sim_rollout = {
        "X": X_rollout,
        "Y": Y_rollout,
        "At": A_rollout,
        # Copy unchanged fields
        "A": sim["A"],
        "z": sim["z"],
        "phi": sim["phi"],
        "s_true": sim["s_true"],
        "true_params": sim["true_params"],
    }
    if "m_train" in sim:
        sim_rollout["m_train"] = sim["m_train"]
    
    return sim_rollout


def compute_test_cascade_metrics(sim_gt, sim_ro, T_train, T_test):
    """
    Compute cascade-level metrics on test period [T_train, T_test).
    
    Args:
        sim_gt: ground truth simulation
        sim_ro: rollout simulation
        T_train: start of test period
        T_test: end of test period (exclusive)
    
    Returns:
        metrics: dict with test metrics
    """
    T, N, D = sim_gt["X"].shape
    T_test = min(T_test, T)  # Ensure within bounds
    
    metrics = {}
    
    # 1. Final cascade size at T_test (per dimension)
    for d in range(D):
        gt_final = sim_gt["X"][T_test-1, :, d].sum()
        ro_final = sim_ro["X"][T_test-1, :, d].sum()
        metrics[f"test_final_size_GT_d{d}"] = int(gt_final)
        metrics[f"test_final_size_RO_d{d}"] = int(ro_final)
        metrics[f"test_final_size_error_d{d}"] = int(abs(gt_final - ro_final))
        metrics[f"test_final_size_ratio_d{d}"] = float(ro_final / max(gt_final, 1))
    
    # 2. Cumulative activation curve RMSE (per dimension)
    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else float("nan")
    
    for d in range(D):
        gt_curve = sim_gt["X"][T_train:T_test, :, d].sum(axis=1)  # (T_test-T_train,)
        ro_curve = sim_ro["X"][T_train:T_test, :, d].sum(axis=1)
        metrics[f"test_cumulative_rmse_d{d}"] = _rmse(gt_curve, ro_curve)
        
        # Correlation
        if len(gt_curve) > 1 and gt_curve.std() > 0 and ro_curve.std() > 0:
            corr = np.corrcoef(gt_curve, ro_curve)[0, 1]
            metrics[f"test_cumulative_corr_d{d}"] = float(corr)
        else:
            metrics[f"test_cumulative_corr_d{d}"] = float("nan")
    
    # 3. First activation time distribution (per dimension)
    def _first_activation_time(X, d, T_start, T_end):
        """Returns first activation time in [T_start, T_end) for each node."""
        first_t = np.full(N, T_end)  # default: never activated
        for t in range(T_start, T_end):
            not_yet = (first_t == T_end)
            newly_active = (X[t, :, d] == 1) & not_yet
            first_t[newly_active] = t
        return first_t
    
    for d in range(D):
        gt_times = _first_activation_time(sim_gt["X"], d, T_train, T_test)
        ro_times = _first_activation_time(sim_ro["X"], d, T_train, T_test)
        
        # Only compare nodes activated in at least one of GT or RO
        activated_in_either = (gt_times < T_test) | (ro_times < T_test)
        
        if activated_in_either.sum() > 0:
            time_diff = np.abs(gt_times[activated_in_either] - ro_times[activated_in_either])
            metrics[f"test_first_activation_mae_d{d}"] = float(time_diff.mean())
            metrics[f"test_first_activation_std_d{d}"] = float(time_diff.std())
            metrics[f"test_first_activation_max_d{d}"] = float(time_diff.max())
            metrics[f"test_n_activated_either_d{d}"] = int(activated_in_either.sum())
        else:
            metrics[f"test_first_activation_mae_d{d}"] = float("nan")
            metrics[f"test_first_activation_std_d{d}"] = float("nan")
            metrics[f"test_first_activation_max_d{d}"] = float("nan")
            metrics[f"test_n_activated_either_d{d}"] = 0
    
    return metrics


def plot_cumulative_activation_split(sim_gt, sim_ro, T_train, T_test, outdir):
    """
    Plot cumulative activation curves: GT vs Rollout, with train/test split markers.
    
    Args:
        sim_gt: ground truth simulation
        sim_ro: rollout simulation
        T_train: train/test split point
        T_test: end of test evaluation window
        outdir: output directory
    """
    T, N, D = sim_gt["X"].shape
    T_test = min(T_test, T)
    
    fig, axes = plt.subplots(1, D, figsize=(5*D, 4.5))
    if D == 1:
        axes = [axes]
    
    for d, ax in enumerate(axes):
        # Compute cumulative activation counts
        gt_curve = sim_gt["X"][:T_test, :, d].sum(axis=1)
        ro_curve = sim_ro["X"][:T_test, :, d].sum(axis=1)
        
        t_range = np.arange(T_test)
        
        # Plot full GT
        ax.plot(t_range, gt_curve, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.8)
        
        # Plot train period (RO = GT) - include T_train point for continuity
        ax.plot(t_range[:T_train+1], ro_curve[:T_train+1], 'b:', linewidth=2.5, 
                label='Train (RO=GT)', alpha=0.7)
        
        # Plot test period (RO diverges) - start from T_train to ensure connection
        ax.plot(t_range[T_train:], ro_curve[T_train:], 'r--', linewidth=2.5, 
                label='Test (Rollout)', alpha=0.8)
        
        # Add split markers
        ax.axvline(T_train, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, 
                   label=f'Train/Test split (t={T_train})')
        
        # Styling
        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel('Cumulative activations', fontsize=12)
        ax.set_title(f'Dimension {d}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, T_test)
        ax.set_ylim(0, max(gt_curve.max(), ro_curve.max()) * 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "test_cumulative_activation_split.png"), dpi=150)
    plt.close()
    
    print(f"[Rollout] Saved cumulative activation plot to {outdir}")
