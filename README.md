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

---

## 8. Example results
```bash
python main_multi_real.py   --real_npz lee_ian_by_cbg_1yr_dr0.005_tr0.6.npz   --max_communities 200   --ref_lon -81.8723   --ref_lat 26.6406   --max_nodes_real 900   --min_nodes_real 100   --num_epochs 300   --lr 5e-4   --label_mode both   --horizon_months 6   --alpha_window 0.9 --lambda_edge 0.5  --save_graphs
Using device: cpu
Loading real data from lee_ian_by_cbg_1yr_dr0.005_tr0.6.npz...
Selected communities (index, N_g, dist_km):
  idx=553, N= 274, dist=0.544
  idx=331, N= 880, dist=0.563
  idx=561, N= 337, dist=0.870
  idx=558, N= 221, dist=1.210
  idx=486, N= 301, dist=1.227
  idx=494, N= 439, dist=1.375
  idx=473, N= 244, dist=1.470
  idx=536, N= 298, dist=1.671
  idx=521, N= 238, dist=1.748
  idx=552, N= 496, dist=1.878
  idx=419, N= 406, dist=1.910
  idx=508, N= 335, dist=2.076
  idx=388, N= 408, dist=2.097
  idx=526, N= 435, dist=2.254
  idx=454, N= 493, dist=2.437
  idx= 71, N= 836, dist=2.627
  idx=474, N= 412, dist=2.714
  idx=336, N= 382, dist=2.749
  idx=497, N= 377, dist=2.766
  idx=394, N= 805, dist=2.770
  idx=296, N= 611, dist=2.848
  idx=298, N= 388, dist=2.855
  idx=442, N= 407, dist=2.903
  idx=514, N= 232, dist=2.996
  idx=520, N= 355, dist=3.085
  idx=161, N= 340, dist=3.505
  idx=222, N= 665, dist=3.574
  idx=487, N= 271, dist=3.596
  idx=543, N= 300, dist=3.684
  idx=475, N= 128, dist=3.713
  idx= 62, N= 644, dist=3.786
  idx=313, N= 458, dist=3.828
  idx=548, N= 474, dist=3.842
  idx=563, N= 155, dist=3.852
  idx=502, N= 351, dist=3.900
  idx=372, N= 856, dist=3.995
  idx= 52, N= 525, dist=4.040
  idx=144, N= 262, dist=4.053
  idx=119, N= 316, dist=4.320
  idx=154, N= 536, dist=4.382
  idx=112, N= 713, dist=4.472
  idx=127, N= 482, dist=4.522
  idx= 68, N= 523, dist=4.535
  idx=233, N= 870, dist=4.554
  idx=499, N= 181, dist=4.590
  idx=524, N= 315, dist=4.886
  idx=476, N= 404, dist=4.994
  idx=282, N= 712, dist=5.017
  idx=550, N= 225, dist=5.097
  idx=461, N= 431, dist=5.126
  idx=353, N= 819, dist=5.189
  idx= 34, N= 247, dist=5.344
  idx=120, N= 701, dist=5.507
  idx= 42, N= 363, dist=5.533
  idx=286, N= 306, dist=5.545
  idx=224, N= 506, dist=5.585
  idx=299, N= 652, dist=5.588
  idx=496, N= 451, dist=5.596
  idx=434, N= 824, dist=5.616
  idx=175, N= 897, dist=5.630
  idx=555, N= 245, dist=5.730
  idx=538, N= 461, dist=5.818
  idx=156, N= 463, dist=6.283
  idx=554, N= 379, dist=6.360
  idx=384, N= 305, dist=6.389
  idx=416, N= 602, dist=6.422
  idx=477, N= 301, dist=6.422
  idx=291, N= 559, dist=6.515
  idx=513, N= 270, dist=6.643
  idx=196, N= 367, dist=6.670
  idx=113, N= 325, dist=6.806
  idx=306, N= 692, dist=6.857
  idx=516, N= 465, dist=6.885
  idx=279, N= 632, dist=6.938
  idx=294, N= 636, dist=7.000
  idx=515, N= 445, dist=7.129
  idx=103, N= 602, dist=7.468
  idx=342, N= 634, dist=7.473
  idx=493, N= 475, dist=7.518
  idx=350, N= 615, dist=7.552
  idx=446, N= 459, dist=7.554
  idx= 57, N= 596, dist=7.630
  idx= 55, N= 652, dist=7.635
  idx=509, N= 294, dist=7.636
  idx=356, N= 506, dist=7.645
  idx=108, N= 817, dist=7.657
  idx=414, N= 612, dist=7.686
  idx=198, N= 501, dist=7.700
  idx=220, N= 406, dist=7.759
  idx=565, N= 338, dist=7.760
  idx=383, N= 638, dist=7.834
  idx= 33, N= 836, dist=7.878
  idx=146, N= 338, dist=7.898
  idx=324, N= 881, dist=7.965
  idx=  6, N= 824, dist=7.997
  idx=292, N= 593, dist=8.129
  idx=547, N= 598, dist=8.157
  idx=406, N= 778, dist=8.194
  idx=116, N= 824, dist=8.257
  idx= 64, N= 894, dist=8.305
  idx=345, N= 525, dist=8.307
  idx=530, N= 527, dist=8.414
  idx=517, N= 456, dist=8.446
  idx=  0, N= 427, dist=8.480
  idx=128, N= 642, dist=8.539
  idx=462, N= 223, dist=8.639
  idx=115, N= 426, dist=8.785
  idx=469, N= 361, dist=8.828
  idx=544, N= 480, dist=8.844
  idx=408, N= 347, dist=8.876
  idx=413, N= 621, dist=8.941
  idx=559, N= 665, dist=9.112
  idx=367, N= 895, dist=9.115
  idx=503, N= 363, dist=9.141
  idx=145, N= 581, dist=9.324
  idx=352, N= 699, dist=9.364
  idx=423, N= 550, dist=9.435
  idx=528, N= 544, dist=9.472
  idx=495, N= 575, dist=9.506
  idx=240, N= 605, dist=9.509
  idx=260, N= 644, dist=9.536
  idx=557, N= 426, dist=9.550
  idx=290, N= 728, dist=9.649
  idx=177, N= 713, dist=9.704
  idx=401, N= 295, dist=9.752
  idx=371, N= 737, dist=9.822
  idx=184, N= 718, dist=9.857
  idx=531, N= 588, dist=9.875
  idx= 53, N= 680, dist=9.933
  idx=432, N= 562, dist=9.993
  idx=317, N= 371, dist=10.058
  idx=349, N= 816, dist=10.134
  idx=568, N= 139, dist=10.178
  idx=194, N= 592, dist=10.195
  idx=218, N= 834, dist=10.196
  idx=433, N= 461, dist=10.391
  idx=297, N= 402, dist=10.392
  idx=188, N= 771, dist=10.397
  idx=186, N= 607, dist=10.398
  idx=190, N= 828, dist=10.531
  idx=512, N= 499, dist=10.598
  idx=379, N= 418, dist=10.612
  idx=251, N= 781, dist=10.642
  idx=409, N= 448, dist=10.656
  idx=376, N= 776, dist=10.698
  idx= 60, N= 513, dist=10.737
  idx=246, N= 703, dist=10.742
  idx=151, N= 760, dist=10.749
  idx=242, N= 370, dist=10.762
  idx=241, N= 740, dist=11.103
  idx=212, N= 618, dist=11.148
  idx=534, N= 384, dist=11.295
  idx=479, N= 566, dist=11.435
  idx=390, N= 560, dist=11.457
  idx= 23, N= 834, dist=11.527
  idx=458, N= 234, dist=11.546
  idx=126, N= 579, dist=11.600
  idx=457, N= 127, dist=11.615
  idx=420, N= 474, dist=11.649
  idx=468, N= 870, dist=11.698
  idx=492, N= 525, dist=11.747
  idx=422, N= 497, dist=11.753
  idx=566, N= 102, dist=11.800
  idx=417, N= 735, dist=11.809
  idx=361, N= 461, dist=11.917
  idx=504, N= 347, dist=11.963
  idx=201, N= 512, dist=12.026
  idx=453, N= 702, dist=12.028
  idx=410, N= 677, dist=12.047
  idx=501, N= 513, dist=12.119
  idx=439, N= 886, dist=12.143
  idx=183, N= 423, dist=12.156
  idx=463, N= 675, dist=12.179
  idx=448, N= 545, dist=12.303
  idx=223, N= 326, dist=12.429
  idx=569, N= 146, dist=12.483
  idx=207, N= 749, dist=12.618
  idx=537, N= 338, dist=12.647
  idx=471, N= 132, dist=12.843
  idx=318, N= 614, dist=12.895
  idx=206, N= 782, dist=12.918
  idx=459, N= 699, dist=12.921
  idx=400, N= 889, dist=12.931
  idx=466, N= 590, dist=12.967
  idx=535, N= 628, dist=12.984
  idx=560, N= 674, dist=13.040
  idx=213, N= 875, dist=13.046
  idx=273, N= 480, dist=13.080
  idx=507, N= 539, dist=13.138
  idx= 84, N= 701, dist=13.172
  idx=283, N= 728, dist=13.222
  idx=522, N= 809, dist=13.363
  idx= 43, N= 342, dist=13.425
  idx= 25, N= 842, dist=13.433
  idx= 80, N= 803, dist=13.586
  idx= 97, N= 751, dist=13.598
  idx=171, N= 299, dist=13.648
  idx=149, N= 738, dist=13.755
  idx=533, N= 757, dist=13.998
  idx=  9, N= 863, dist=14.020
#communities after filtering = 200
[INFO] Real-data meta: T=13, T_train=7, T_val=6
[INFO] Real-data meta: T=13, T_train=7, T_val=6
[INFO] pos_weight per type (sell, repair, vacate) (capped): [14.135292838425986, 20.26526564680407, 46.52741127550511]
[REAL-NEURAL epoch 001] loss=0.669 (nll=2.495, edge=0.019)
[REAL-NEURAL epoch 005] loss=0.431 (nll=1.632, edge=0.017)
[REAL-NEURAL epoch 010] loss=0.402 (nll=1.540, edge=0.017)
[REAL-NEURAL epoch 015] loss=0.388 (nll=1.493, edge=0.017)
[REAL-NEURAL epoch 020] loss=0.383 (nll=1.482, edge=0.017)
[REAL-NEURAL epoch 025] loss=0.385 (nll=1.497, edge=0.017)
[REAL-NEURAL epoch 030] loss=0.373 (nll=1.446, edge=0.017)
[REAL-NEURAL epoch 035] loss=0.369 (nll=1.440, edge=0.017)
[REAL-NEURAL epoch 040] loss=0.371 (nll=1.466, edge=0.017)
[REAL-NEURAL epoch 045] loss=0.363 (nll=1.426, edge=0.017)
[REAL-NEURAL epoch 050] loss=0.367 (nll=1.457, edge=0.017)
[REAL-NEURAL epoch 055] loss=0.354 (nll=1.395, edge=0.017)
[REAL-NEURAL epoch 060] loss=0.354 (nll=1.397, edge=0.017)
[REAL-NEURAL epoch 065] loss=0.356 (nll=1.390, edge=0.017)
[REAL-NEURAL epoch 070] loss=0.350 (nll=1.382, edge=0.017)
[REAL-NEURAL epoch 075] loss=0.353 (nll=1.393, edge=0.017)
[REAL-NEURAL epoch 080] loss=0.354 (nll=1.414, edge=0.017)
[REAL-NEURAL epoch 085] loss=0.343 (nll=1.362, edge=0.017)
[REAL-NEURAL epoch 090] loss=0.343 (nll=1.356, edge=0.017)
[REAL-NEURAL epoch 095] loss=0.342 (nll=1.364, edge=0.017)
[REAL-NEURAL epoch 100] loss=0.344 (nll=1.378, edge=0.017)
[REAL-NEURAL epoch 105] loss=0.341 (nll=1.362, edge=0.017)
[REAL-NEURAL epoch 110] loss=0.339 (nll=1.369, edge=0.017)
[REAL-NEURAL epoch 115] loss=0.334 (nll=1.340, edge=0.017)
[REAL-NEURAL epoch 120] loss=0.336 (nll=1.357, edge=0.017)
[REAL-NEURAL epoch 125] loss=0.334 (nll=1.344, edge=0.017)
[REAL-NEURAL epoch 130] loss=0.333 (nll=1.339, edge=0.017)
[REAL-NEURAL epoch 135] loss=0.337 (nll=1.369, edge=0.017)
[REAL-NEURAL epoch 140] loss=0.334 (nll=1.370, edge=0.017)
[REAL-NEURAL epoch 145] loss=0.328 (nll=1.318, edge=0.017)
[REAL-NEURAL epoch 150] loss=0.329 (nll=1.334, edge=0.017)
[REAL-NEURAL epoch 155] loss=0.328 (nll=1.328, edge=0.017)
[REAL-NEURAL epoch 160] loss=0.329 (nll=1.347, edge=0.017)
[REAL-NEURAL epoch 165] loss=0.321 (nll=1.308, edge=0.017)
[REAL-NEURAL epoch 170] loss=0.321 (nll=1.321, edge=0.017)
[REAL-NEURAL epoch 175] loss=0.321 (nll=1.306, edge=0.017)
[REAL-NEURAL epoch 180] loss=0.317 (nll=1.296, edge=0.017)
[REAL-NEURAL epoch 185] loss=0.319 (nll=1.309, edge=0.017)
[REAL-NEURAL epoch 190] loss=0.318 (nll=1.296, edge=0.017)
[REAL-NEURAL epoch 195] loss=0.315 (nll=1.297, edge=0.017)
[REAL-NEURAL epoch 200] loss=0.314 (nll=1.292, edge=0.017)
[REAL-NEURAL epoch 205] loss=0.321 (nll=1.313, edge=0.017)
[REAL-NEURAL epoch 210] loss=0.316 (nll=1.317, edge=0.017)
[REAL-NEURAL epoch 215] loss=0.315 (nll=1.291, edge=0.017)
[REAL-NEURAL epoch 220] loss=0.319 (nll=1.323, edge=0.017)
[REAL-NEURAL epoch 225] loss=0.312 (nll=1.299, edge=0.017)
[REAL-NEURAL epoch 230] loss=0.311 (nll=1.286, edge=0.017)
[REAL-NEURAL epoch 235] loss=0.313 (nll=1.287, edge=0.017)
[REAL-NEURAL epoch 240] loss=0.315 (nll=1.293, edge=0.017)
[REAL-NEURAL epoch 245] loss=0.310 (nll=1.291, edge=0.017)
[REAL-NEURAL epoch 250] loss=0.313 (nll=1.299, edge=0.017)
[REAL-NEURAL epoch 255] loss=0.312 (nll=1.284, edge=0.017)
[REAL-NEURAL epoch 260] loss=0.308 (nll=1.289, edge=0.017)
[REAL-NEURAL epoch 265] loss=0.307 (nll=1.270, edge=0.017)
[REAL-NEURAL epoch 270] loss=0.306 (nll=1.279, edge=0.017)
[REAL-NEURAL epoch 275] loss=0.307 (nll=1.274, edge=0.017)
[REAL-NEURAL epoch 280] loss=0.310 (nll=1.285, edge=0.017)
[REAL-NEURAL epoch 285] loss=0.305 (nll=1.276, edge=0.017)
[REAL-NEURAL epoch 290] loss=0.303 (nll=1.262, edge=0.017)
[REAL-NEURAL epoch 295] loss=0.305 (nll=1.275, edge=0.017)
[REAL-NEURAL epoch 300] loss=0.306 (nll=1.262, edge=0.017)

[Sanity check] Tail event counts over selected communities:
    sell  - exact_tail_count = 3237, window_tail_count (H=6) = 5992
    repair- exact_tail_count = 1042, window_tail_count (H=6) = 2264
    vacate- exact_tail_count = 0, window_tail_count (H=6) = 0
[INFO] Saved learned graphs for 200 communities to lee_ian_by_cbg_1yr_dr0.005_tr0.6_learned_graphs.npz

[Real-data coupled Hawkes results]
T_train = 7, T_val = 6
#communities (trained) = 200
    sell_auc_tail:          0.7775738328579003
    sell_ap_tail:           0.04650519445691261
    sell_auc_window_tail:  0.7478070021191713
    sell_ap_window_tail:   0.04200633710204722
  repair_auc_tail:          0.9795812464262316
  repair_ap_tail:           0.06418420127241092
  repair_auc_window_tail:  0.9777129510620689
  repair_ap_window_tail:   0.1047360045921198
  ```
