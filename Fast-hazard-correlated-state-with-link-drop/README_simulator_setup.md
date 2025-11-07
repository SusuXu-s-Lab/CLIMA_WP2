# Controlling “Slow” Activation in the Simulator

This README explains how to produce **delayed cascades**—i.e., runs where **no (or almost no) activations happen in the first few steps**—using the time–gating knobs implemented in `simulate_icD_dynamic(...)`. It also shows CLI usage (e.g., via `viz_eval_nn_ic.py`) and how to visualize the dynamics.

---

## Why slow activation?

In many experiments you want to avoid immediate activity so you can:

- Measure early-time calibration without contamination from initial spikes  
- Observe how diffusion behaves once a warm-up period ends  
- Keep layouts stable in animations before deletions kick in (when using the “drop-on-d=1” rule)

We implement slow activation by **biasing the logits** of both the self and edge hazards at early timesteps.

---

## How it works (concept)

For step \(t\), we add a time-dependent **logit bias** \(b(t)\) to the baseline hazards:

- **Self channel** (per node \(i\), state \(d\)):
  \[
  s_{t,i,d}=\sigma(s^{	ext{base}}_{i,d}) + b_s(t))
  \]
- **Edge channel** (per attempt \(j\!	o\!i\), state pair \(k\!	o\!d\)):
  \[
  q_{t,(j\cdot i,k\cdot d)}=\sigma(c_0 + c_1\phi_{j\cdot i} + B_{d,k} + b_e(t))
  \]

Choose \(b_s(t)\), \(b_e(t)\) **strongly negative** during early steps to suppress activations. After the warm-up, the bias goes to zero and hazards revert to their base levels.

---

## Simulator knobs

These flags exist in `simulate_icD_dynamic(...)` and are exposed by CLI wrappers (e.g., `viz_eval_nn_ic.py`):

- `--K_self` (int, default `0`)  
  Suppress **self** activations before this step. `0` disables gating.  
  Example: `--K_self 3` ≈ no self‐driven events for `t < 3`.

- `--K_edge` (int, default `0`)  
  Suppress **edge** activations (contagion attempts) before this step. `0` disables gating.  
  Example: `--K_edge 3` ≈ no contagion‐driven events for `t < 3`.

- `--bias_self` (float, default `10.0`)  
  Magnitude of the **negative** logit bias used while self gating is active. Larger ⇒ stronger early suppression.  
  Typical: `10–20`.

- `--bias_edge` (float, default `10.0`)  
  Same as above, for the edge channel.

- `--mode_self` (`hard|linear`, default `hard`)  
  Time–gate shape for self:  
  - `hard`: \(b_s(t)=-	ext{bias}\) for \(t<K_{	ext{self}}\) else \(0\).  
  - `linear`: ramps from \(-	ext{bias}\) at \(t=0\) to \(0\) at \(t=K_{	ext{self}}\).

- `--mode_edge` (`hard|linear`, default `hard`)  
  Time–gate shape for edge (same behavior as above).

- `--p_seed` (float)  
  Probability a node is seeded **at \(t=0\)**. For a truly quiet start, set `--p_seed 0.0`.

> **Tip:** Use **both** `K_self` and `K_edge` when you want *no* early activations from any source.

---

## Quick recipes

### 1) Strict “no activation” before \(t=3\)

```bash
python viz_eval_nn_ic.py   --N 60 --T 150 --D 3   --p_edge 0.10 --p_seed 0.00   --K_self 3 --K_edge 3   --mode_self hard --mode_edge hard   --bias_self 20 --bias_edge 20   --em_iters 24 --epochs_self 10 --epochs_edge 5   --lr_self 3e-3 --lr_edge 5e-3   --seed 7 --outdir results_slow_hard --make_plots 1
```

### 2) Soft ramp to normal by \(t=3\)

```bash
python viz_eval_nn_ic.py   --N 60 --T 150 --D 3   --p_edge 0.10 --p_seed 0.00   --K_self 3 --K_edge 3   --mode_self linear --mode_edge linear   --bias_self 10 --bias_edge 10   --em_iters 24 --epochs_self 10 --epochs_edge 5   --lr_self 3e-3 --lr_edge 5e-3   --seed 7 --outdir results_slow_linear --make_plots 1
```

### 3) Delay contagion only (allow rare self seeds)

```bash
python viz_eval_nn_ic.py   ...   --p_seed 0.01   --K_self 0   --K_edge 3 --mode_edge hard --bias_edge 15
```

---

## Interactions & best practices

- **Dynamic edge deletion (drop-on-d=1).**  
  If you use `--drop_dim 1` (links are removed after both endpoints activate on state-1), deletions won’t begin until after the warm-up since nothing fires early. This keeps the early frames stable and makes animations clearer.

- **Training stability under slow starts.**  
  Early steps yield few or zero **attempts**, so the edge net sees little signal initially. To compensate:  
  - Increase EM iterations (e.g., `--em_iters +2~4`),  
  - Add a couple of epochs for the self net per EM step (`--epochs_self +2`) to stabilize responsibilities,  
  - If you use pairwise ranking loss, **anneal** its weight during the first few EM rounds.

- **Metrics during warm-up.**  
  With near-zero positives, calibration and threshold metrics can be skewed. Either:  
  - Report both **all-time** and **post-warm-up** metrics, or  
  - Exclude \(t<\max(K_{	ext{self}},K_{	ext{edge}})\) when computing event-prediction metrics.

- **Calibration (ECE/MCE).**  
  Strong gating creates many probabilities near 0 early; **adaptive ECE/MCE** (equal-count bins) stabilizes tail estimates.

---

## Visualizing slow cascades

Use the animation helper to confirm the delay and see when the cascade takes off:

```bash
python visualize_dynamic_network.py   --load_npz path/to/sim.npz   --outdir sim_viz_slow   --drop_dim 1   --step 1 --fps 8
```

You should see **no colored nodes** (or very few if `p_seed>0`) before your chosen \(K\), followed by a rise after the warm-up.

---

## Troubleshooting

- **I still see early activations.**  
  Ensure `--p_seed 0.0`. If a few remain, increase `--bias_self`/`--bias_edge` (e.g., to 20–25) or switch to `--mode_self hard` / `--mode_edge hard`.

- **Cascade never takes off.**  
  Post–warm-up hazards may be too small. Reduce `bias_*`, shorten \(K\), or slightly increase base hazards (e.g., tune `a0/a1` for self, `c0/c1/B` for edge).

- **Edge metrics degrade with slow starts.**  
  That’s usually because there are fewer early attempts. Add a few EM iterations and edge epochs; you can also delay or shrink ranking-weight early on.

---

## Reference: Architecture knobs (for context)

- `--self_hidden` (int, default: 64) — width of self MLP hidden layers.  
- `--self_depth` (int, default: 2) — number of hidden layers in self MLP.  
- `--edge_emb` (int, default: 8) — embedding size for categorical edge context (e.g., destination/source states).  
- `--edge_hidden` (int, default: 96) — width of edge MLP hidden layers.  
- `--edge_depth` (int, default: 3) — number of hidden layers in edge MLP.

> **Tuning tip:** If `B_R2` is low while NLL/Brier look fine, the edge net may be underpowered—bump `edge_hidden` or `edge_depth` a notch and add a touch of weight decay (`--wd_edge`).
