### 1  Household Features Generation  

| Synthesised column | Generation logic | Alignment rationale |
|--------------------|------------------|---------------------|
| `income` | Community-level Beta distribution → household-level Gaussian noise | Captures income spatial heterogeneity; matches “time-varying individual features” |
| `building_value`  | High-value communities (15 %) get higher Normal mean | Reflects property value clusters after disaster |
| `damage_level`    | Categorical {0, .25, .5, .75, 1}; biased by “high-damage” communities | Provides initial physical impact state used by decision process |
| `population_scaled` | Uniform integer [1–7] → scaled to 0-1 | Normalised household size (node feature) |
| `age`             | Communities tagged *old / young / mixed* → draw from Normal centres | Gives demographic variance required by `demo_i(t)` |
| `race`            | Numeric {0, .25, .5, 1}; 80 % of communities White-dominant | Encodes categorical race as continuous value for similarity metric |

**Community baselines → household noise**  
All continuous attributes are first set at the community level to impose spatial autocorrelation (as assumed in Eq. 18 similarity), then perturbed with small Gaussian noise so each household remains unique.

**Damage & high-value zoning**  
15 % of geohash-based communities are randomly flagged as *high-damage* or *high-value* to mimic disaster hotspots and affluent clusters, aligning with the formulation’s need for heterogeneous initial conditions \(s_i(0)\) and structural priors \(π_1^0, π_2^0\).

**Irreversible decision compatibility**  
`damage_level` directly influences the initial probability of `repair_state` and neighbourhood vacancy / sales bias, ensuring downstream state diffusion complies with irreversible rule (Assumption 1 in Section 2.4).

**Numeric encoding**  
All features are normalised to \([0,1]\) or discrete numeric levels so they can be fed directly into neural components \( \text{NN}_\text{self}, \text{NN}_\text{influence}, \text{NN}_\text{form}\).

```bash
Output → DataFrame columns:
['home', 'community', 'building_value', 'income',
 'damage_level', 'population_scaled', 'age', 'race']
```
