### Household Features Generation  

| Synthesised column | Generation logic (concise but explicit) |
|--------------------|------------------------------------------|
| **`income`** | Geohash communities are ordered; the first half use **Beta(2, 2)**, the rest **Beta(0.5, 2)** as a community-level centre. Each household draws `Normal(centre, 0.1)` then clips to [0, 1]. |
| **`building_value`** | Randomly flag **15 %** of communities as *high-value* → centre = 0.7; others centre = 0.5. Household value = `Normal(centre, 0.05)` clipped 0–1. |
| **`damage_level`** | Mark **15 %** communities as *high-damage*. Sample categorical \{0, 0.25, 0.5, 0.75, 1\} with probs **[0.1, 0.1, 0.2, 0.3, 0.3]** if high-damage else **[0.7, 0.1, 0.1, 0.05, 0.05]**. |
| **`population_scaled`** | Draw household size ∈ {1,…,7} uniformly, divide by 7 → range 0–1. |
| **`age`** | Each community is labelled **old / young / mixed** with probs 0.3 / 0.3 / 0.4. Household age ~ `Normal(0.8, 0.05)` (old), `Normal(0.2, 0.05)` (young) or `Normal(0.5, 0.1)` (mixed); clip 0–1. |
| **`race`** | Numeric encoding {0 = White, 0.25 = Black, 0.5 = Hispanic, 1 = Asian}. **80 %** communities are White-dominant (0); the rest choose a random non-white dominant. For each household, sample race with weight **0.7** on the dominant code and **0.1** on the others. |

All features are scaled/encoded in \([0,1]\) or discrete numeric levels.

```bash
Output → DataFrame columns:
['home', 'community', 'building_value', 'income',
 'damage_level', 'population_scaled', 'age', 'race']
```

### T = 0 Household-State Generation

| State column       | Generation rule (deterministic parameters shown) |
|--------------------|--------------------------------------------------|
| **repair_state**   | Allowed **only if `damage_level` > 0**. Activation probability: `p = 0.2 + 0.6 × damage` (capped at 1). Sample Bernoulli for each household. |
| **vacancy_state**  | Compute community-level mean damage `d̄_c`. Probability: `p = 0.03 + 0.20 × d̄_c` (max 0.30). |
| **sales_state**    | Same damage driver but steeper: `p = 0.05 + 0.30 × d̄_c` (max 0.30). |

All three states obey **irreversibility**: once set to 1 at time `t = 0`, they remain 1 in all future steps.  
For `t > 0`, states are pre-filled as zero; they will be updated dynamically by the FR-SIC diffusion model during simulation.

```python
# Output columns:
['home', 'time', 'repair_state', 'vacancy_state', 'sales_state']
```

### Similarity Matrix Construction

Pairwise household similarity is computed using the exponential kernel defined in Equation (18) of the formulation:

```text
similarity(i, j, t) = exp( - (‖demo_i - demo_j‖² / σ_demo² + dist_ij² / σ_geo²) )
```

This reflects the assumption that closer demographic and spatial proximity leads to higher likelihood of bonding-type links.

The construction proceeds as follows:

| Step | Description |
|------|-------------|
| **1** | Extract demographic features: `income`, `age`, `race`. Compute pairwise Euclidean distance → `demo_dist`. |
| **2** | Decode the 8-digit geohash to latitude/longitude and compute geodesic distance matrix using Haversine formula → `geo_dist` (in meters). |
| **3** | Compute normalisation constants `σ_demo` and `σ_geo` as the median of all pairwise distances (excluding diagonals). |
| **4** | Apply exponential kernel elementwise to get final similarity matrix (values in (0,1)). Returned as a symmetric DataFrame indexed by `home`. |

This similarity matrix is static across time in the current setup.

```python
# Output: similarity[i][j] = similarity[j][i] ∈ (0,1)
DataFrame shape: (N_households × N_households)
```


### Interaction Potential Matrix Construction

Pairwise interaction potential between households is computed following Equation (19) in the formulation. This reflects the likelihood of **bridging link formation** based on feature similarity, state alignment, and spatial proximity.

The model uses a linearized surrogate for `NN_form`, combining demographic distance, social state, and geographic distance:

```
interaction_potential(i, j, t) = sigmoid( wᵀ · [f_ij(t), s_i(t), s_j(t), dist_ij] )
```

| Step | Description |
|------|-------------|
| **1** | Extract absolute demographic differences: `f_ij(t)`` = |demo_i - demo_j|`, where `demo` includes `income`, `age`, `race`. Shape: *(N, N, 3)* |
| **2** | Extract household state vectors `s_i(t)` and `s_j(t)` for all nodes at time `t`. Shape: *(N, N, 3)* for each. |
| **3** | Decode geohash and compute true geodesic distance (meters) between each pair → `dist_ij`, reshaped to *(N, N, 1)*. |
| **4** | Concatenate feature vectors for each household pair:  
  `[f_ij, s_i, s_j, dist_ij]` → total length = 10. |
| **5** | Apply fixed linear weights over the feature vector:  
    - Strong negative weights on dissimilarity and distance  
    - Mild negative weights on state mismatch  
    - Final score passed through `sigmoid` to get probability in (0, 1) |

The resulting interaction potential matrix is symmetric and dynamic—recomputed at each time step `t` based on the current node states.

```python
# Output: interaction_potential[i][j] ∈ (0,1)
DataFrame shape: (N_households × N_households)
```

