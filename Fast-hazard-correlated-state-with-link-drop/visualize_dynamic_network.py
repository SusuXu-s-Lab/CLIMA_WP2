# visualize_dynamic_network.py
# Visualize dynamic IC networks with D-dimensional states and drop-on-d rule.
# - Static: initial/final graphs, rasters, active-over-time, degree histograms
# - Animated: per-dimension GIFs + optional 3-panel GIF (uses every frame by default)
# Usage:
#   python visualize_dynamic_network.py --load_npz path/to/sim.npz --outdir sim_viz --drop_dim 1 --step 1 --fps 8
# Or import and call visualize_network_dynamics(sim, outdir=..., step=1)

import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


def build_A_sequence(A0, X, drop_dim=1):
    """
    Reconstruct time-varying adjacency A_seq[t] (used from t->t+1) from:
      - A0: initial directed adjacency (N,N)
      - X:  cumulative states (T,N,D)
    Deletion rule: if both endpoints are active on drop_dim at time t,
    delete the link in BOTH directions for all future times.
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
                A_curr[np.ix_(idx, idx)] = 0  # undirected removal
            A_seq[t] = A_curr
    else:
        for t in range(1, T):
            A_seq[t] = A_curr
    return A_seq


def _nx_layout(A, seed=0):
    """Undirected layout for readability."""
    if not HAS_NX:
        return None, None
    G = nx.from_numpy_array(A, create_using=nx.DiGraph()).to_undirected()
    pos = nx.spring_layout(G, seed=seed)
    return G, pos


def _draw_graph(A_t, node_vals=None, title="", pos=None, show_removed=None, cmap="viridis"):
    """
    Draw a single frame:
      - A_t: adjacency at time t (N,N)
      - node_vals: color/size by activation in {0,1} (N,) or None
      - show_removed: boolean mask (N,N) of edges removed vs initial (optional)
    """
    if HAS_NX and pos is not None:
        G = nx.from_numpy_array(A_t, create_using=nx.DiGraph()).to_undirected()
        plt.title(title)
        # Highlight removed edges (relative to A0) in faint red
        if show_removed is not None and np.any(show_removed):
            G0 = nx.from_numpy_array(show_removed.astype(int), create_using=nx.DiGraph()).to_undirected()
            nx.draw_networkx_edges(G0, pos, edge_color="#D55E00", width=0.8, alpha=0.25)
        nx.draw_networkx_edges(G, pos, edge_color="#7f7f7f", width=0.8, alpha=0.6)
        if node_vals is None:
            nx.draw_networkx_nodes(G, pos, node_size=160, node_color="#4477aa")
        else:
            vals = np.asarray(node_vals).ravel()
            sizes = 180 + 420 * vals
            nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=vals,
                                           cmap=cmap, vmin=0, vmax=1)
            cb = plt.colorbar(nodes, shrink=0.85)
            cb.set_label("Active (0/1)")
        plt.axis("off")
    else:
        # Fallback: adjacency heatmap + node bar
        plt.subplot(1, 2, 1)
        plt.imshow(A_t, interpolation="nearest", cmap="gray_r")
        plt.title(title or "Adjacency"); plt.axis("off")
        plt.subplot(1, 2, 2)
        if node_vals is not None:
            plt.imshow(node_vals[None, :], interpolation="nearest",
                       cmap=cmap, vmin=0, vmax=1, aspect="auto")
            plt.title("Node state"); plt.yticks([])
        else:
            plt.text(0.5, 0.5, "No node values", ha="center", va="center")
            plt.axis("off")


def visualize_network_dynamics(
    sim,
    outdir="sim_viz",
    drop_dim=1,
    three_panel=True,
    per_dim=True,
    step=1,
    fps=8,
    layout_seed=0
):
    """
    Create static figures + GIF animations of dynamic network:
      - Uses sim["A"] (initial adjacency), sim["X"] (T,N,D).
      - If sim["At"] exists, uses it; else reconstructs A_seq via the deletion rule.
      - Saves figures/GIFs into `outdir` and returns a dict of written paths.
    """
    os.makedirs(outdir, exist_ok=True)
    A0 = sim["A"]
    X = sim["X"]  # (T,N,D)
    T, N, D = X.shape

    # Dynamic adjacency
    A_seq = sim.get("At", None)
    if A_seq is None:
        A_seq = build_A_sequence(A0, X, drop_dim=drop_dim)

    # Static: initial/final networks and rasters
    _, pos = _nx_layout(A0, seed=layout_seed)

    # Initial network
    plt.figure(figsize=(6, 5))
    _draw_graph(A0, node_vals=None, title="Initial network (A0)", pos=pos)
    p_init = os.path.join(outdir, "network_initial.png")
    plt.tight_layout(); plt.savefig(p_init, dpi=140); plt.close()

    # Final network with node color/size by final activation per dim
    A_final = A_seq[-1]
    plt.figure(figsize=(6, 5))
    _draw_graph(A_final, node_vals=None, title="Final network", pos=pos)
    p_final = os.path.join(outdir, "network_final.png")
    plt.tight_layout(); plt.savefig(p_final, dpi=140); plt.close()

    per_dim_final = []
    for d in range(D):
        plt.figure(figsize=(6, 5))
        _draw_graph(A_final, node_vals=X[-1, :, d], title=f"Final network — dim {d}", pos=pos)
        p = os.path.join(outdir, f"network_final_dim{d}.png")
        plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()
        per_dim_final.append(p)

    # Rasters + counts
    raster_paths = []
    for d in range(D):
        plt.figure(figsize=(8, 4))
        plt.imshow(X[:, :, d].T, aspect='auto', interpolation='nearest', cmap='Greys')
        plt.xlabel("Time"); plt.ylabel("Node index"); plt.title(f"Activation raster (dim {d})")
        rp = os.path.join(outdir, f"raster_dim{d}.png")
        plt.tight_layout(); plt.savefig(rp, dpi=140); plt.close()
        raster_paths.append(rp)

    plt.figure(figsize=(7, 4))
    for d in range(D):
        plt.plot(np.arange(T), X[:, :, d].sum(axis=1), label=f"dim {d}")
    plt.xlabel("Time"); plt.ylabel("# active nodes"); plt.title("Active nodes over time")
    plt.legend(); plt.tight_layout()
    counts_path = os.path.join(outdir, "active_over_time.png")
    plt.savefig(counts_path, dpi=140); plt.close()

    # Degree histograms (initial vs final, undirected)
    def _deg_hist(A, title, fname):
        if HAS_NX:
            G = nx.from_numpy_array(A, create_using=nx.DiGraph()).to_undirected()
            degs = np.array([d for _, d in G.degree()])
        else:
            degs = (A + A.T).astype(bool).sum(axis=1)
        plt.figure(figsize=(6, 4))
        bins = min(20, max(5, int(np.sqrt(len(degs)))))
        plt.hist(degs, bins=bins)
        plt.xlabel("Degree"); plt.ylabel("Count"); plt.title(title)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=140); plt.close()

    _deg_hist(A0, "Initial degree histogram", "initial_degree_histogram.png")
    _deg_hist(A_final, "Final degree histogram", "final_degree_histogram.png")

    # Animations (GIF). Show removed edges faintly red. Use EVERY frame by default.
    gifs = {}
    if not HAS_IMAGEIO:
        print("[viz] imageio not installed; skipping GIF creation.")
    else:
        s = max(1, int(step))
        removed_masks = [(A0 > 0) & (A_seq[t] == 0) for t in range(T)]

        if per_dim:
            for d in range(D):
                frames = []
                tmp_dir = Path(outdir) / f"anim_dim{d}_frames"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                for t in range(0, T, s):
                    plt.figure(figsize=(6, 5))
                    _draw_graph(
                        A_seq[t],
                        node_vals=X[t, :, d],
                        title=f"t={t}  dim={d}  (drop_dim={drop_dim})",
                        pos=pos,
                        show_removed=removed_masks[t]
                    )
                    f = tmp_dir / f"frame_{t:04d}.png"
                    plt.tight_layout(); plt.savefig(f, dpi=140); plt.close()
                    frames.append(f)
                gif_path = Path(outdir) / f"anim_dim{d}.gif"
                with imageio.get_writer(gif_path, mode="I", duration=1.0/max(fps, 1)) as wr:
                    for f in frames:
                        wr.append_data(imageio.imread(f))
                gifs[f"dim{d}"] = str(gif_path)

        if three_panel and D >= 2:
            frames = []
            tmp_dir = Path(outdir) / "anim_3dims_frames"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for t in range(0, T, s):
                plt.figure(figsize=(14, 4.8))
                for d in range(D):
                    plt.subplot(1, D, d + 1)
                    _draw_graph(
                        A_seq[t],
                        node_vals=X[t, :, d],
                        title=f"t={t}  dim {d}",
                        pos=pos,
                        show_removed=removed_masks[t]
                    )
                plt.suptitle(f"Dynamic network (drop_dim={drop_dim})")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                f = tmp_dir / f"frame_{t:04d}.png"
                plt.savefig(f, dpi=140); plt.close()
                frames.append(f)
            gif_path = Path(outdir) / "anim_3dims.gif"
            with imageio.get_writer(gif_path, mode="I", duration=1.0/max(fps, 1)) as wr:
                for f in frames:
                    wr.append_data(imageio.imread(f))
            gifs["three_panel"] = str(gif_path)

    return {
        "network_initial": p_init,
        "network_final": p_final,
        "per_dim_final": per_dim_final,
        "rasters": raster_paths,
        "counts": counts_path,
        "gifs": gifs,
    }


def _main():
    ap = argparse.ArgumentParser("Visualize dynamic network from saved npz (A, X).")
    ap.add_argument("--load_npz", type=str, required=True,
                    help="Path to npz with A (N,N) and X (T,N,D)")
    ap.add_argument("--outdir", type=str, default="sim_viz")
    ap.add_argument("--drop_dim", type=int, default=1)
    ap.add_argument("--step", type=int, default=1, help="frame stride (1 = every frame)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--three_panel", type=int, default=1)
    ap.add_argument("--per_dim", type=int, default=1)
    ap.add_argument("--layout_seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.load_npz)
    if "A" not in data or "X" not in data:
        raise ValueError("npz must contain 'A' (N,N) and 'X' (T,N,D).")

    sim = dict(A=data["A"], X=data["X"])
    paths = visualize_network_dynamics(
        sim,
        outdir=args.outdir,
        drop_dim=args.drop_dim,
        three_panel=bool(args.three_panel),
        per_dim=bool(args.per_dim),
        step=args.step,
        fps=args.fps,
        layout_seed=args.layout_seed
    )
    print("Saved outputs:", paths)


if __name__ == "__main__":
    _main()
