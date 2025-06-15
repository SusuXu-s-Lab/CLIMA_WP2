### 1  Household Features Generation  

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
