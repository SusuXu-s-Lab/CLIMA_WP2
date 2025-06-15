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


### Interaction Potential Matrix Construction for each T

Interaction potential between households is computed using a linearised version of the formulation's Equation (19), combining demographic difference, social state vectors, and geodesic distance.

The final interaction score is:

```
interaction_potential(i, j, t) = sigmoid( w · [f_ij, s_i, s_j, dist_ij] )
```

| Step | Description |
|------|-------------|
| **1** | Compute `f_ij(t)` = absolute difference in `income`, `age`, `race` between household `i` and `j`. Result: shape (N, N, 3). |
| **2** | Extract `s_i(t)` and `s_j(t)` from current `repair_state`, `vacancy_state`, `sales_state`. Each becomes shape (N, N, 3). |
| **3** | Decode 8-digit geohash and calculate haversine distance between locations → `dist_ij` (in meters). Reshape to (N, N, 1). |
| **4** | Concatenate `[f_ij, s_i, s_j, dist_ij]` into one 10-dimensional vector for each pair. |
| **5** | Apply fixed linear weights, followed by sigmoid function to produce interaction potential values in (0, 1). |

Output is a symmetric (N × N) DataFrame. This matrix is recomputed at each time step `t` using time-varying states `s(t)`.

```python
weights = np.array([-2.0, -1.0, -1.0,     # f_ij part
                    -1.0, -1.0, -1.0,     # s_i part
                    -1.0, -1.0, -1.0,     # s_j part
                    -10.0])               # dist_ij
# Output format:
DataFrame: interaction_potential[i][j] in (0,1)
```

### T = 0 Link Matrix Generation

The initial social network \( G_0 \) is generated using a **softmax-based categorical sampling** strategy as described in Equation (13)–(14) of the formulation. Each unordered household pair \((i, j)\) has a chance to form one of three link types:

- `0`: no link  
- `1`: bonding link (demographic-based affinity)  
- `2`: bridging link (interaction-based potential)  

---

The link assignment follows this procedure:

| Step | Description |
|------|-------------|
| **1** | Iterate all unordered household pairs \((i < j)\), extract their `similarity(i, j)` and `interaction_potential(i, j)` values. |
| **2** | Construct softmax logits `[1.0, α₀ × similarity, β₀ × interaction]`. Here, `1.0` is the base score for no-link, and α₀, β₀ are configurable weights (default: 0.9 and 0.5). |
| **3** | Apply softmax over the 3 logits to obtain `[p₀₀, p₀₁, p₀₂]`, the probabilities of assigning no-link, bonding, or bridging. |
| **4** | Sample link type using a categorical distribution with these probabilities. |
| **5** | Fill a symmetric \(N × N\) matrix where entry `(i,j)` and `(j,i)` are set to the sampled link type. |

The result is a symmetric adjacency matrix representing the initial state of the network, where links emerge probabilistically according to household similarity and interaction potential.

```python
# Output: link_matrix[i][j] ∈ {0, 1, 2}
DataFrame shape: (N_households × N_households)
```

This initialization aligns with the model’s probabilistic link formation assumption at \(t=0\), serving as the foundation for subsequent link evolution governed by Eq. (15)–(17).


### T > 0 Link Transition

At each timestep \( t > 0 \), the link matrix \( G_t \) evolves from the previous matrix \( G_{t-1} \) based on the diffusion-aware transition model described in Equations (13)–(17) of the formulation.

The transition logic depends on:
- Node similarity (demographic and spatial)
- Pairwise interaction potential
- Dynamic node states (particularly `vacancy_state`)

---

#### Link Transition Rules

| Previous Link Type \( \ell_{ij}(t{-}1) \) | Transition Mechanism |
|------------------------------------------|-----------------------|
| **0 (no link)**      | Use softmax over logits: `[1, α × similarity(i,j), β × interaction(i,j)]` to sample from \{0, 1, 2\}. |
| **1 (bonding)**      | Always preserved: \( p_{11} = 1 \). |
| **2 (bridging)**     | If both `i` and `j` are not vacant → retain with probability `similarity(i,j)`.  
Otherwise decay using `γ × similarity(i,j)`. Sample from \{0, 2\}. |

---

### Implementation Steps

| Step | Description |
|------|-------------|
| **1** | Extract `similarity(i, j)` and `interaction_potential(i, j)` matrices from time `t`. |
| **2** | Extract `vacancy_state` from `house_states` at time `t`. |
| **3** | For each unordered pair \((i, j)\), determine prior link type \(\ell_{ij}(t{-}1)\) and apply the corresponding transition rule. |
| **4** | Sample next link type using either softmax or Bernoulli as needed, and update both \((i, j)\) and \((j, i)\) entries in the link matrix. |

The full transition logic is implemented in `update_link_matrix_one_step(...)`, and the resulting matrix is symmetric and aligned with the evolving household states.

```python
# Output: link_matrix[i][j] ∈ {0, 1, 2}
DataFrame shape: (N_households × N_households)
```

This transition step ensures that the structure of the network co-evolves with household-level dynamics, enabling link formation, persistence, and decay consistent with the assumptions in the paper.
