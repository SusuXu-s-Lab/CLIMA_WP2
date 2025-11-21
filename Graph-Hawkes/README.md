# Graph-Hawkes for Parcel-Level Hazard Recovery (Real Data)

This repository implements a **graph-based neural Hawkes model** to predict multi-type parcel-level events (e.g., *sell*, *repair*, *vacate*) after a major hazard.  

The “real-data” pipeline is currently set up for **Hurricane Ian / Lee County, FL**, using:

- Parcel-level data (`fl_lee.csv`, Regrid + county assessor),
- Sales transactions (`sales_data.csv`),
- Repair permits (`repair_data.csv`),
- Hazard intensity / damage map (`hurricane_ian_damage_map.csv`).

The workflow is:

1. **Prepare real data** → build parcel- and CBG-level tensors and save as `.npz`.
2. **Train + evaluate Graph Hawkes model** on that `.npz`.

---

## 1. Environment & Dependencies

Tested with Python 3.7+.

Core dependencies (install with `pip`):

```bash
pip install numpy pandas scipy torch scikit-learn
```

If you run into import errors, also check for:

- `argparse` (standard library),
- `collections`, `math`, etc. (standard library).

---

## 2. Repository Structure (relevant files)

- `prepare_real_ian_by_cbg.py`  
  Preprocesses parcel + event data into a **community-level** dataset stored in a `.npz` file.

- `main_multi_real.py`  
  Main entry point to **train and evaluate** the coupled Hawkes model on real data.

- `train_eval_multi_real.py`  
  Training and evaluation code (called by `main_multi_real.py`).

- `model_multi_real.py`  
  Defines the graph-based Hawkes model for multi-type events.

You’ll mainly interact with:

1. `prepare_real_ian_by_cbg.py`
2. `main_multi_real.py`

---

## 3. Step 1 – Preparing the Real Dataset

### 3.1 Required input files

Assumed directory layout (you can change paths):

```text
hurricane_ian_data/
  ├── fl_lee.csv                   # Parcel-level data (Regrid + county assessor)
  ├── sales_data.csv               # Sales events
  ├── repair_data.csv              # Repair / permit events
  └── hurricane_ian_damage_map.csv # Hazard / damage points
```

### 3.2 Basic command

Example (with long horizon end date):

```bash
python prepare_real_ian_by_cbg.py \
  --parcel_csv hurricane_ian_data/fl_lee.csv \
  --sales_csv  hurricane_ian_data/sales_data.csv \
  --repair_csv hurricane_ian_data/repair_data.csv \
  --damage_csv hurricane_ian_data/hurricane_ian_damage_map.csv \
  --map_radius_deg 0.0007 \
  --damage_radius_deg 0.005 \
  --out_npz lee_ian_by_cbg_1yr.npz \
  --train_ratio 0.6 \
  --end_date 2023-09-01
```

This will produce something like:

- `lee_ian_by_cbg_test.npz`
- `lee_ian_by_cbg_test_dr0.005_tr0.8.npz`  ← **recommended to use as `--real_npz`**

You’ll see logs like:

- Total events per type (sell, repair, vacate)
- Time index info (start date, end date, \(T\))
- Per-month event counts
- Train/val event counts and 6-month “window” totals

These logs are important sanity checks.

### 3.3 Key arguments (data prep)

#### Required file paths

- `--parcel_csv`  
  Path to the parcel-level table (e.g., `fl_lee.csv`).

- `--sales_csv`  
  Sales events. Mapped to parcels via `parcelnumb` and/or spatial join.

- `--repair_csv`  
  Repair / permit events. Mapped to parcels similarly.

- `--damage_csv`  
  Damage map / hazard intensity points (e.g., flood depth, estimated loss).  
  Used to associate each parcel with nearby damage points.

#### Spatial radii

- `--map_radius_deg` (float, default ~`0.0007`)  
  Radius (in degrees) for matching **point-based events** (sales, repairs) to parcels.  
  Roughly \(\mathcal{O}(100\ \text{m})\) at these latitudes.  
  - *Smaller*: stricter mapping, fewer events matched.  
  - *Larger*: more tolerant mapping, but risk of wrong parcel association.

- `--damage_radius_deg` (float, default `0.005`)  
  Radius for associating **damage map points** to parcels.  
  - *Smaller*: only heavily damaged parcels get non-zero hazard.  
  - *Larger*: spreads hazard signal more broadly.

#### Temporal settings

- `--ian_date` (default `2022-09-28`)  
  Hurricane landfall date. Used as a reference for “post-Ian” filtering.

- `--end_date`  
  Final date for the time index.  
  - Example `2025-11-01` → \(T = 39\) monthly bins from 2022-09-01 to 2025-11-01.  
  - Example `2024-10-01` → \(T = 26\) bins.

  **Important:** Because `usps_vacancy_date` is concentrated at specific dates  
  (e.g., 2023-03-01 and 2025-11-01), choosing an `end_date` that stops *before*  
  those later updates will heavily affect how many vacate events appear in your tail.

#### Train/validation split

- `--train_ratio` (float, default `0.8`)  
  Fraction of **time steps** used for training.  
  - With \(T=39\), `train_ratio=0.8` ⇒ \(T_{\text{train}} = 31\), \(T_{\text{val}} = 8\).  
  - With \(T=13\), `train_ratio=0.6` ⇒ \(T_{\text{train}} = 7\), \(T_{\text{val}} = 6\).

  **Trade-off:**

  - Higher train ratio → more history to learn from, but shorter validation tail.
  - Lower train ratio → less training history, but longer and more challenging tail.

#### Community filtering

- `--min_cbg_events` (int, default `1`)  
  Minimum **total events** (sell+repair+vacate) required to keep a Census Block Group (CBG) as a “community”.  
  Increase this if you want to focus only on active communities.

---

## 4. Step 2 – Training the Graph Hawkes Model

### 4.1 Basic command

The preprocessed `.npz` file is fed into `main_multi_real.py`.

A configuration that gave **good-looking metrics** in your last run was:

```bash
python main_multi_real.py \
  --real_npz lee_ian_by_cbg_1yr_dr0.005_tr0.6.npz \
  --max_communities 200 \
  --ref_lon -81.8723 \
  --ref_lat 26.6406 \
  --max_nodes_real 500 \
  --min_nodes_real 100 \
  --num_epochs 200 \
  --lr 5e-4 \
  --label_mode both \
  --horizon_months 6 \
  --alpha_window 0.9 \
  --lambda_edge 0.5
```

This produced logs like:

- Edge loss non-zero (because `lambda_edge=0.5`)
- Good AUC/AP for sell and repair, vacate mostly neutral (no tail vacate events in that split).

### 4.2 Key arguments (model / training)

#### Core data & selection

- `--real_npz`  
  Path to prepared `.npz` (the parameter-labeled one is recommended,  
  e.g. `lee_ian_by_cbg_test_dr0.005_tr0.6.npz`).

- `--max_communities` (int, default `200`)  
  Maximum number of CBG communities to train on, **closest** to a reference point.  
  The script prints:

  ```text
  Selected communities (index, N_g, dist_km):
    idx=553, N=274, dist=0.544
    ...
  #communities after filtering = 123
  ```

  Tuning:

  - Increase if you want broader geographic coverage (more communities).
  - Decrease if you want to prioritize speed and near-epicenter behavior.

- `--ref_lon`, `--ref_lat`  
  Reference location (lon/lat) used to compute distance (km) from each community centroid.  
  Communities are sorted by this distance before applying `max_communities`.  
  For Lee County, `(-81.8723, 26.6406)` is a reasonable central location.

- `--max_nodes_real`, `--min_nodes_real`  
  Per-community node constraints (parcels per CBG).  

  - `max_nodes_real=500`: cap extremely dense CBGs for computational tractability.  
  - `min_nodes_real=100`: drop tiny CBGs with very few parcels.

  If you want to include small communities, **lower** `min_nodes_real`.  
  If memory is an issue, **lower** `max_nodes_real`.

#### Optimization

- `--num_epochs` (int, default `100`)  
  Number of training epochs.

  Tuning:

  - If training loss is still dropping significantly: increase.
  - If training loss is low but validation metrics degrade: try fewer epochs or more regularization.

- `--batch_size` (int, default `32`)  
  Number of communities per batch in training.  
  Larger batch size improves stability but requires more memory.

- `--lr` (float, default `5e-2`)  
  Learning rate for Adam.

  In practice, the following have worked well:

  - `5e-3` and `5e-4` are often more stable than the default `5e-2`.

  Tuning:

  - If loss is noisy or diverging: **decrease** `lr`.
  - If training is very slow but stable: you can cautiously **increase** `lr`.

- `--weight_decay` (float)  
  L2 regularization on model weights. Increase slightly if you see overfitting.

- `--seed`  
  Random seed for reproducibility.

- `--device` (`cpu` / `cuda`)  
  Training device. Use `cuda` if you have a GPU.

#### Labeling & horizons

- `--label_mode` (default `"both"`)  
  How to construct labels for the prediction task:

  - `"exact"`: only events in the **exact horizon month**.
  - `"window"`: any event within the **horizon window** (e.g., next \(H\) months).
  - `"both"`: combines exact-month and window labels.

  For your experiments you used: `--label_mode both`.

- `--horizon_months` (int, default `6`)  
  Prediction horizon \(H\) in months.

  - Example: `horizon_months=6` → label if an event occurs between \(t+1\) and \(t+6\).

- `--alpha_window` (float in \([0,1]\), default maybe `0.5`)  
  Weight that mixes window-based vs. exact-month objectives when `label_mode="both"`.

  Rough interpretation:

  - \(\alpha_\text{window} \approx 0\) → focus on **exact month** tail prediction.
  - \(\alpha_\text{window} \approx 1\) → focus on **H-month window** tail prediction.

  You’ve tried:

  - `alpha_window=0.5`: balanced.
  - `alpha_window=0.9`: strongly emphasize window-tail classification.

  Tuning:

  - If you care about **any event in the next \(H\) months**: push `alpha_window` toward 1.
  - If you care about **exact timing**: push it toward 0.

#### Graph sparsity / edge regularization

- `--lambda_edge` (float \(\ge 0\), default `0.0`)  
  L1 penalty on the inferred community-to-community influence matrix.  

  - `lambda_edge=0.0` → **no** sparsity penalty, edge loss stays `0.000` in logs.
  - `lambda_edge>0` → encourages sparser adjacency; edge loss becomes non-zero:

    ```text
    [REAL-NEURAL epoch 001] loss=0.748 (nll=2.791, edge=0.027)
    ...
    ```

  Tuning:

  - If the learned graph is too dense / noisy: increase `lambda_edge`.
  - If performance drops too much: decrease `lambda_edge`.

#### Other internals

Inside `main_multi_real.py` and `train_eval_multi_real.py`, there are additional settings (not all exposed via CLI):

- Hidden dimension sizes (`d_hid`, etc.).
- Kernel decay parameters (e.g., per-event-type Hawkes decay rates).
- `k_tops` (top-\(k\) neighbors per layer / head in the graph module).

You can expose these via new CLI arguments if you want to experiment more deeply, but for normal usage the defaults are a reasonable starting point.

---

## 5. Understanding the Logs & Sanity Checks

When you run `prepare_real_ian_by_cbg.py`, you see diagnostics like:

```text
Total events per type in Y (sell, repair, vacate): [117358, 12901, 11354]
T_train=31, T_val=8 (train_ratio=0.8)

[SANITY] Exact events per type (sell, repair, vacate):
    Train: [102267, 9095, 3504]
    Val  : [15091, 3806, 7850]
    Total: [117358, 12901, 11354]

[SANITY] Window-6-month events per type (sell, repair, vacate):
    Train window: [574596, 54570, 21024]
    Val window  : [119999, 14364, 7850]
    Total window: [694595, 68934, 28874]
```

And per-time counts:

```text
[PER-TIME] Events per month (date, t, sell, repair, vacate):
2022-09-01  t=00 (train)  [102, 0, 0]
2022-10-01  t=01 (train)  [3225, 979, 0]
...
2023-03-01  t=06 (train)  [5185, 931, 3503]
...
2025-11-01  t=38 (val)    [0, 0, 7850]
```

Use these to verify:

- Vacate events are indeed sharply concentrated (e.g., 2023-03-01 and 2025-11-01), which explains why changing `end_date` can zero out tail vacate labels.
- Repairs tend to cluster at certain months (e.g., 2025-06 to 2025-09 in your longer horizon).

During training, you see logs along the lines of:

```text
[INFO] Real-data meta: T=13, T_train=7, T_val=6
[INFO] pos_weight per type (sell, repair, vacate) (capped): [14.42, 17.40, 41.68]

[REAL-NEURAL epoch 200] loss=0.327 (nll=1.400, edge=0.023)

[Sanity check] Tail event counts over selected communities:
    sell  - exact_tail_count = 1370, window_tail_count (H=6) = 2543
    repair- exact_tail_count = 508, window_tail_count (H=6) = 964

[Real-data coupled Hawkes results]
T_train = 7, T_val = 6
#communities (trained) = 123
    sell_auc_tail:          0.7513
    sell_ap_tail:           0.0290
    sell_auc_window_tail:   0.7210
    sell_ap_window_tail:    0.0449
  repair_auc_tail:          0.9692
  repair_ap_tail:           0.0387
  repair_auc_window_tail:   0.9699
  repair_ap_window_tail:    0.0791
```

Interpretation:

- `pos_weight per type` balances rare events (e.g., vacate) in the loss; extremely large weights are capped (e.g., at 1000).
- Tail counts confirm how many positive labels actually exist in the **validation horizon** for each event type.
  - Here, vacate has **no** tail events in the chosen split → AUC \(\approx 0.5\) and AP \(\approx 0.0\) is expected.
- `sell` and `repair` metrics are meaningful; you can compare across hyperparameters.

---

## 6. Practical Tuning Tips

1. **Choose `end_date` to capture the signal you care about.**  
   If USPS vacancy updates are only at a couple of dates (e.g., 2023-03-01 and 2025-11-01),  
   and you want vacate events in the tail, make sure your validation window includes one of those spikes.

2. **Adjust `train_ratio` to balance history vs tail.**

   - If tail performance is unstable: try a slightly larger validation window (smaller `train_ratio`).

3. **Start with conservative optimization hyperparameters.**

   - `lr = 5 \times 10^{-4}` or `5 \times 10^{-3}`  
   - `num_epochs = 150–200`  
   - `lambda_edge = 0.5` (for a reasonably sparse graph)

4. **Watch the edge loss.**

   - If `edge` term is near zero even with non-zero `lambda_edge`, the model may be ignoring the penalty  
     (e.g., weights already very small). Decrease `lambda_edge` if performance drops, or increase if graph is still too dense.

5. **For rare event types (vacate):**

   - Check **tail event counts** in logs: if zero, metrics are not informative.
   - To study vacate meaningfully, you may need:
     - Different `end_date` / `train_ratio`, or
     - A different labelling approach (e.g., earlier USPS vacancy dates) in the pre-processing script.

---

## 7. Example End-to-End Run

```bash
# 1. Prepare data (Horizon until 2025-11-01, 60% of time for training)
python prepare_real_ian_by_cbg.py \
  --parcel_csv hurricane_ian_data/fl_lee.csv \
  --sales_csv  hurricane_ian_data/sales_data.csv \
  --repair_csv hurricane_ian_data/repair_data.csv \
  --damage_csv hurricane_ian_data/hurricane_ian_damage_map.csv \
  --map_radius_deg 0.0007 \
  --damage_radius_deg 0.005 \
  --out_npz lee_ian_by_cbg_1yr.npz \
  --train_ratio 0.6 \
  --end_date 2023-09-01

# 2. Train & evaluate graph Hawkes model
python main_multi_real.py \
  --real_npz lee_ian_by_cbg_1yr_dr0.005_tr0.6.npz \
  --max_communities 200 \
  --ref_lon -81.8723 \
  --ref_lat 26.6406 \
  --max_nodes_real 500 \
  --min_nodes_real 100 \
  --num_epochs 200 \
  --lr 5e-4 \
  --label_mode both \
  --horizon_months 6 \
  --alpha_window 0.9 \
  --lambda_edge 0.5
```
