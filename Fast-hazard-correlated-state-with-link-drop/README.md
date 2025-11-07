# Neural Independent Cascade + EM (Joint Self & Edge) — with Dynamic Link Deletion

This repo implements a **Neural Independent Cascade (IC)** model with an **EM** trainer that:
- supports **D-dimensional states** per node (jointly modeled),
- uses **dynamic adjacency** with a rule that **removes an edge when both endpoints are active on a specified dimension (`drop_dim`)**,
- parameterizes **self-activation** via a **joint MLP** and **edge diffusion** via a **joint, monotone-in-φ MLP**,
- includes two optional nudges:
  1. a **`both_d1` flag** as an extra input to the edge MLP to model imminent link deletion when both endpoints are “on” in a given dimension;
  2. **responsibility annealing** in the E-step (τ-schedule and edge-floor schedule).
---

## Files (key)

- `viz_eval_nn_ic.py` — simulation, training, full evaluation, and basic plots.
- `em_trainer_nn_ic.py` — EM trainer (E-step + M-step) with annealing and the pairwise rank regularizer.
- `models_nn_ic.py` — joint self MLP and joint edge MLP (monotone in φ), with the `both_d1` flag and φ standardization.

---

## Quick start

Example (the balanced configuration that works well with dynamic link deletion on `d=1`):

```bash
python viz_eval_nn_ic.py \
  --N 60 --T 150 --D 3 \
  --p_edge 0.10 --p_seed 0.02 --rho 0.4 --drop_dim 1 \
  --self_hidden 64 --self_depth 2 --edge_emb 8 --edge_hidden 96 --edge_depth 3 \
  --em_iters 24 --epochs_self 10 --epochs_edge 8 \
  --lr_self 3e-3 --lr_edge 8e-3 --wd_edge 3e-3 \
  --tau_resp 0.85 --edge_resp_floor 0.12 \
  --lambda_rank 1.0 --rank_pairs 1024 --rank_margin 0.02 \
  --self_weighting sqrt_invfreq \
  --outdir results_dynamic_drop --make_plots 1
```

Optional annealing / φ-warmup (if you want responsibility schedules and stabilized φ stats in early iterations):
```bash
  --tau_start 0.90 --tau_end 0.85 \
  --edge_floor_start 0.08 --edge_floor_end 0.12 \
  --phi_warmup 2
```

Optional controling simulator on how slow the hazard propagates:
```bash
--p_seed 0.0 --K_self 3 --K_edge 3 --mode_self linear --mode_edge linear --bias_self 10 --bias_edge 10
```

Visualize the network -- the viz_eval_nn_ic.py file also uses this function to automatically visualize network dynamics
```bash
python visualize_dynamic_network.py \
  --load_npz path/to/sim.npz \
  --outdir sim_viz \
  --drop_dim 1 \
  --step 1 \
  --fps 8
```

My output (see the explaination about each metric below):
```bash
Self_RMSE: 0.0527
Self_R2: 0.7672
B_RMSE: 0.0872
B_R2: 0.5483
Attr_RMSE_vsTruePosterior: 0.1121
Attr_R2_vsTruePosterior: 0.9350

NLL: 0.2023
Brier: 0.0554
ECE: 0.0147
MCE: 0.3048
ECE_adaptive: 0.0196
MCE_adaptive: 0.1181
ROC_AUC: 0.8463
PR_AUC: 0.5046

ACC_t0p5: 0.9274
F1_t0p5: 0.2439
Precision_t0p5: 0.7812
Recall_t0p5: 0.1445
Specificity_t0p5: 0.9964
BalancedAcc_t0p5: 0.5705
Thr_Youden: 0.1206
ACC_tY: 0.8486
F1_tY: 0.4242
Precision_tY: 0.3067
Recall_tY: 0.6879
Specificity_tY: 0.8628
BalancedAcc_tY: 0.7753
Thr_bestF1: 0.2281
F1_bestF1: 0.5121
ACC_tF1: 0.9152
Precision_tF1: 0.4798
Recall_tF1: 0.5491
Specificity_tF1: 0.9475
BalancedAcc_tF1: 0.7483
Thr_recallTarget: 0.0551
RecallTarget: 0.8000
ACC_tR: 0.6954
F1_tR: 0.2996
Precision_tR: 0.1841
Recall_tR: 0.8035
Specificity_tR: 0.6859
BalancedAcc_tR: 0.7447

drop_dim: 1
self_weighting: sqrt_invfreq
tau_resp: 0.8500
edge_resp_floor: 0.1200
lambda_rank: 1.0000
Self_RMSE_d0: 0.0471
Self_R2_d0: 0.8055
Self_RMSE_d1: 0.0437
Self_R2_d1: 0.2482
Self_RMSE_d2: 0.0649
Self_R2_d2: 0.7867
---
```

## Parameter reference

### Simulation (in `viz_eval_nn_ic.py`)

- `--N` (int): Number of nodes.
- `--T` (int): Number of time steps.
- `--D` (int): Number of **state dimensions** per node.
- `--p_edge` (float): Prior probability of an edge in the initial graph.
- `--p_seed` (float): Probability a node/dimension is initially active at `t=0`.
- `--rho` (float): Correlation among the D features in the latent per-node vector `z ~ N(0, Σρ)` with `Σρ = (1-ρ)I + ρ 11ᵀ`.
- `--drop_dim` (int): **If both endpoints are active on `drop_dim` at time `t`, that undirected link is *removed* at `t+1`**. (Implements “people leave the community” effect.) If you don't want to drop the dimension, set drop_dim=-1.
- `--seed` (int): RNG seed.
- `--c0`,`--c1` (floats): Constants controlling the *true* edge attempt probability via `σ(c0 + c1 φ_ji + B[d,k])` used in simulation (ground truth).

### Training (common)
- `--self_hidden` (int, default: 64):
Width of each hidden layer in the self-activation MLP. Larger values increase capacity to model nonlinear self hazards from node features (and past states, if you pass them).
When to increase: underfitting (high Self_RMSE / low Self_R2).
When to decrease: overfitting (train ≪ val), or if you need faster EM steps.
Typical range: 32–128.
- `--self_depth` (int, default: 2):
Number of hidden layers (not counting the output layer) in the self-activation MLP. Deeper models capture richer interactions; shallow models are faster and stabler when data are sparse.
When to increase: complex/self-heterogeneous nodes or strong feature interactions.
When to decrease: small datasets or unstable EM.
Typical range: 1–3.
- `--edge_emb` (int, default: 8):
Size of the embedding vectors used for categorical edge context, e.g., destination state ddd and source state kkk (and any other discrete tags you include). These embeddings let the edge net share strength across (d,k)(d,k)(d,k) pairs while still learning pair-specific behavior.
When to increase: many state types (large DDD) or strong (d,k)(d,k)(d,k)-specific effects.
When to decrease: tiny DDD, very sparse attempts, or to reduce memory/compute.
Typical range: 4–16.
- `--edge_hidden` (int, default: 96):
Width of each hidden layer in the edge-activation MLP (the per-attempt success head). Governs capacity to model the mapping from ϕ\phiϕ, embeddings, and context to qj→i(k→d)q_{j\to i}^{(k\to d)}qj→i(k→d)​.
When to increase: low B_R2 / clear underfitting of edge probabilities.
When to decrease: overfitting of rare attempts or slow M-steps.
Typical range: 64–192.
- `--edge_depth` (int, default: 3):
Number of hidden layers (not counting the output) in the edge MLP. More depth helps capture nonlinearities in ϕ\phiϕ and cross-dim interactions (via the embeddings), but can destabilize training if data are scarce.
When to increase: complex diffusion patterns; strong benefit from ranking loss.
When to decrease: sparse cascades, noisy responsibilities, or if calibration degrades.
Typical range: 2–4.

- `--em_iters` (int): Number of EM iterations. Each iteration performs one E-step and separate M-steps for self and edge nets.
- `--epochs_self` (int): Number of gradient epochs for the self MLP per EM iteration.
- `--epochs_edge` (int): Number of gradient epochs for the edge MLP per EM iteration.
- `--lr_self`, `--lr_edge` (floats): Learning rates for the self / edge optimizers.
- `--wd_self`, `--wd_edge` (floats): Weight decay (L2) for each optimizer.
- `--phi_monotone` (0/1): If 1, the edge head is **constrained to be monotone non-decreasing in standardized φ** via a non-negative slope `softplus(w_φ)`. Improves identifiability and stability.
- `--device` (`cpu` or `cuda`): Where to run the models.

### E-step stabilizers (fixed per-iteration)

- `--tau_resp` (float): **Responsibility sharpening** (τ). Values <1 push responsibility mass toward the single most plausible cause; values >1 smooth them. We use **τ≈0.85** to gently sharpen.
- `--edge_resp_floor` (float): After computing responsibilities, enforce a **minimum total edge responsibility** when `y=1`. Prevents the E-step from attributing *all* activations to the self-channel, especially when edges are sparse.

### Responsibility annealing (optional schedules)

If you pass these, they override the fixed `--tau_resp` and `--edge_resp_floor`, linearly interpolated across EM iterations:

- `--tau_start`, `--tau_end` (floats): Start/end τ for **E-step sharpening** schedule.
- `--edge_floor_start`, `--edge_floor_end` (floats): Start/end for **edge responsibility floor** schedule.

**Typical pattern:** Start with higher τ (smoother, e.g. 0.90) and lower edge floor (e.g. 0.08) to fit the self channel, then anneal toward sharper τ (0.85) and slightly higher floor (0.12) so edges pick up residuals.

### φ-statistics warmup

- `--phi_warmup` (int): For the first K EM iterations, recompute mean/var of φ **from attempted edges** and update the edge net’s internal standardization. Helps early stability under changing active-edge sets.

### Pairwise ranking regularizer (edge head)

- `--lambda_rank` (float): Weight of the pairwise margin loss (see `_edge_pairwise_rank_loss` below). Encourages **monotone ordering** of predicted edge probabilities w.r.t. φ **within each (dest dim d, source dim k)** bucket.
- `--rank_pairs` (int): Number of random pairs sampled per M-step to compute the ranking loss.
- `--rank_margin` (float): Margin for the hinge loss; larger margin demands a stronger separation between predictions at higher vs lower φ.

### Per-state weighting (self head)

- `--self_weighting` (`none | invfreq | sqrt_invfreq`): Reweights the self head’s per-sample BCE using the (posterior) frequency of positives for each dimension `d`.
  - `invfreq`: strong upweighting of rare dims; can be unstable if extremely rare.
  - `sqrt_invfreq`: gentler upweighting; robust default for dynamic deletion on one dimension.

---

## What is `_edge_pairwise_rank_loss`?

**Goal:** reinforce the model’s **ordering** of edge attempt probabilities with respect to the scalar influence `φ`, **holding context fixed to the same (d, k) bucket** (destination dimension `d` and source dimension `k`). The idea: if `φ_lo < φ_hi` for two attempts that share `(d,k)`, the predicted edge probability should satisfy `q(φ_hi) ≥ q(φ_lo) + margin`.

**How it’s computed (per M-step for the edge head):**

1. **Bucket** the edge training rows by `(d, k)`.
2. For each bucket, sample pairs `(lo, hi)` with `φ_lo < φ_hi`.
3. Compute the edge head’s predictions `q_lo`, `q_hi` **for the same destination dim `d`**.
4. Apply a **hinge**: `L_rank = mean( max( margin - (q_hi - q_lo), 0 ) )`.
5. Add to the edge loss: `L_edge_total = BCE_edge + λ_rank * L_rank`.

This **does not** replace the BCE fit; it supplements it with a shape prior that stabilizes training under sparse, dynamic edges. If λ is too large, it can distort **calibration** (hurting B_R²) while still improving ranking metrics (AUC/PR). In practice, the values you found (`lambda_rank=1.0`, `rank_pairs≈1k`, `margin≈0.02`) balance shape and scale reasonably well for the dynamic-drop simulator.

---

## High-level EM loop (what happens each iteration)

1. **E-step (responsibility update)**  
   For every candidate activation `(t,i,d)` with label `y∈{0,1}`:
   - Compute self probability `s = self_net(z_i, x_i_prev)[d]`.
   - For each new neighbor attempt `(j,k)` (newly active neighbor `j` at `t-1` in dim `k`):
     - Build the **edge context** `(φ_ji, k, z_i, x_i_prev, z_j, x_j_prev, both_d1)` and get `q_{j,k→d}` via the edge net.
   - Combine by **noisy-OR**: `p = 1 - (1 - s) ∏(1 - q_{j,k→d})`.
   - If `y=0`: target responsibility for self and all attempts is `0`.
   - If `y=1`: compute posteriors
     - `r_self ∝ s ∏(1 - q)`  
     - `r_m ∝ (1 - s) · q_m ∏_{n≠m}(1 - q_n)`  
     normalize by `p`.  
     Then **sharpen** via τ and enforce **edge responsibility floor** if requested.

2. **M-step (self net)**  
   - Construct supervised rows `(z_i, x_i_prev, d, target=r_self)`.
   - Fit the self MLP with BCE. Optionally reweight per dimension via `--self_weighting`.

3. **M-step (edge net)**  
   - Construct supervised rows `(φ_ji, k, z_i, x_i_prev, z_j, x_j_prev, both_d1, d, target=r_edge)`.
   - Fit the edge MLP with BCE, plus optional **pairwise ranking**.
   - φ **standardization** is internal; set once or re-warmed for the first few iterations.

Repeat for `--em_iters` iterations.

---

## Metrics (what to watch)

- **Mechanism fidelity**  
  - `Self_RMSE, Self_R2`: how well the self-head recovers true self probabilities across nodes & dims.
  - `B_RMSE, B_R2`: how well the edge-head matches true **per-attempt** edge probabilities on actual attempts.
  - `Attr_RMSE, Attr_R2`: how close responsibility attribution (self vs network) is to the true posterior at events (`y=1`).

- **Event prediction**  
  - `NLL, Brier, ECE/MCE (and adaptive variants)`: calibration-quality of overall event probabilities.
  - `ROC_AUC, PR_AUC`: ranking under class imbalance.
  - Thresholded metrics at 0.5, **Youden’s J**, **best-F1**, and **recall-target**: pick the operating point that matches your application.

**Note:** The “drop-on-d1” rule reduces supervision for edges exactly where activation is common, which can depress `B_R2`. Your baseline setup balances this well.

---

## Tips

- Capacity vs. stability: Start with moderate widths/depths. If B_R2 is poor while calibration (NLL/Brier) is okay, the edge net may be underpowered—grow edge_hidden or edge_depth a notch. If EM oscillates, back off depth first.
- Embeddings: For D=3D{=}3D=3, --edge_emb 8 is usually enough. For larger DDD, scale roughly like min⁡(16,  2D)\min(16,\;2D)min(16,2D).
- Regularization: Pair bigger models with a bit more --wd_self / --wd_edge. If you use the monotone + ranking loss, deeper/wider edge nets often need slightly higher weight decay to keep calibration steady.
- Compute scaling: Roughly O(depth×hidden2)O(\text{depth} \times \text{hidden}^2)O(depth×hidden2) per forward; edge M-step cost also scales with #attempts. Increase batch size or sub-sample pairs in the ranking loss if steps slow down.
- For **mechanism recovery**, keep `tau≈0.85` and moderate `edge_resp_floor≈0.12` (your baseline), avoid extreme λ for ranking, and consider short φ warmup.
- For **better detection metrics**, you can anneal responsibilities to shift mass to edges, but watch `B_R2` and Self_R²—they may drop.
- Always report thresholded metrics at **best-F1** or **recall-target**, not only at 0.5.
