### 1  Household Features Generation  

| Synthesised column | Generation logic | Alignment rationale |
|--------------------|------------------|---------------------|
| `income` | Community-level Beta distribution → household-level Gaussian noise | Captures income spatial heterogeneity; matches “time-varying individual features” |
| `building_value`  | High-value communities (15 %) get higher Normal mean | Reflects property value clusters after disaster |
| `damage_level`    | Categorical {0, .25, .5, .75, 1}; biased by “high-damage” communities | Provides initial physical impact state used by decision process |
| `population_scaled` | Uniform integer [1–7] → scaled to 0-1 | Normalised household size (node feature) |
| `age`             | Communities tagged *old / young / mixed* → draw from Normal centres | Gives demographic variance|
| `race`            | Numeric {0, .25, .5, 1}; 80 % of communities White-dominant | Encodes categorical race as continuous value for similarity metric |

```bash
Output → DataFrame columns:
['home', 'community', 'building_value', 'income',
 'damage_level', 'population_scaled', 'age', 'race']
```
