### Household Features Generation

The `generate_household_features()` function enriches a given household-level DataFrame with synthetic but geographically structured demographic and environmental attributes. Each household is identified by an 8-digit geohash (`home`), from which community labels are extracted using the first 6 characters.

This function simulates six core features for each household:

#### Community-Based Structure
All households are grouped by their 6-digit geohash prefixes into communities. Many features are generated at the **community level first** (e.g., damage probability, income center) and then assigned to households with small variations.

#### `damage_level`
- Categorical variable with 5 levels: `{0.0, 0.25, 0.5, 0.75, 1.0}`
- Certain communities are marked as high-damage zones with higher probabilities for severe damage.
- Sampling is biased based on whether a household's community is in a high-damage region.

#### `income`
- Continuous variable in \([0, 1]\)
- Spatially heterogeneous: the first half of communities are assigned higher income centers via a Beta distribution.
- Household income values are sampled around their community center with some variance.

#### `building_value`
- Continuous variable in \([0, 1]\)
- 15% of communities are randomly selected as “high-value” zones.
- Each household's value is sampled from a Normal distribution around its community center (with smaller variance than income).

#### `population_scaled`
- Simulated household size (1 to 7 members), scaled into the range \([0, 1]\)
- Uniformly sampled without geographic dependency.

#### `age`
- Continuous variable in \([0, 1]\) representing the age of the household's decision maker.
- Communities are assigned one of three types: `'old'`, `'young'`, or `'mixed'`, which control the center of the household age distribution.

#### `race`
- Discrete value ∈ `{0.0, 0.25, 0.5, 1.0}` representing racial group membership:
  - `0.0` = White, `0.25` = Black, `0.5` = Hispanic, `1.0` = Asian
- 80% of communities are dominated by White households (`0.0`).
- Within each community, household race is sampled using a biased distribution favoring the dominant group.
